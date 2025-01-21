import logging
import os
import json
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from phi.agent import Agent
from phi.model.openai import OpenAILike

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
load_dotenv()

BATCH_SIZE = 10  # Process 10 paragraphs concurrently


async def score_paragraphs_batch(
        session: aiohttp.ClientSession,
        paragraphs: List[str],
        query: str,
        base_url: str,
        api_key: str
) -> List[float]:
    """Score a batch of paragraphs concurrently."""

    async def score_single(paragraph: str) -> float:
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system",
                     "content": "You are a document analyzer. Given a paragraph and query, analyze their relevance. Respond only with a score from 0.0 to 1.0."},
                    {"role": "user",
                     "content": f"Task: Score how relevant this paragraph is to the query. Score from 0.0 (irrelevant) to 1.0 (highly relevant).\n\nQuery: {query}\nParagraph: {paragraph}\n\nRespond only with the numerical score:"}
                ],
                "temperature": 0.1
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return min(max(float(result['choices'][0]['message']['content'].strip()), 0.0), 1.0)
                return 0.0
        except Exception as e:
            logger.error(f"Error scoring paragraph: {str(e)}")
            return 0.0

    return await asyncio.gather(*(score_single(p) for p in paragraphs))


def extract_relevant_paragraphs(query: str, filenames: Optional[List[str]] = None) -> str:
    """Wrapper function for async implementation."""
    return asyncio.run(extract_relevant_paragraphs_async(query, filenames))


async def extract_relevant_paragraphs_async(query: str, filenames: Optional[List[str]] = None) -> str:
    """Extract relevant paragraphs using parallel processing."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(project_root, "documents")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    max_paragraphs = int(os.getenv("MAX_PARAGRAPHS", "5"))
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
    api_key = os.getenv("OPENAI_API_KEY")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )

    if not os.path.exists(docs_path):
        logger.warning(f"Documents directory not found. Creating: {docs_path}")
        os.makedirs(docs_path)

    def load_document(file_path: str) -> Optional[List[str]]:
        """Load and split a PDF document into paragraphs."""
        logger.info(f"Loading document: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            full_text = " ".join(page.page_content for page in pages)
            paragraphs = text_splitter.split_text(full_text)
            logger.info(f"Successfully loaded document with {len(paragraphs)} paragraphs")
            return paragraphs
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return None

    try:
        if filenames is None:
            filenames = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]

        if not filenames:
            logger.warning("No PDF files found to analyze")
            return json.dumps({})

        results: Dict[str, List[Dict[str, Any]]] = {}
        total_paragraphs = 0
        relevant_paragraphs = 0

        async with aiohttp.ClientSession() as session:
            for filename in filenames:
                file_path = os.path.join(docs_path, filename)
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {filename}")
                    continue

                paragraphs = load_document(file_path)
                if not paragraphs:
                    continue

                total_paragraphs += len(paragraphs)
                scored_paragraphs = []

                # Process in batches
                for chunk in [paragraphs[i:i + BATCH_SIZE] for i in range(0, len(paragraphs), BATCH_SIZE)]:
                    scores = await score_paragraphs_batch(session, chunk, query, base_url, api_key)
                    relevant = [(p, s) for p, s in zip(chunk, scores) if s > relevance_threshold]
                    scored_paragraphs.extend({"text": p, "relevance_score": round(s, 4)} for p, s in relevant)
                    relevant_paragraphs += len(relevant)

                if scored_paragraphs:
                    scored_paragraphs.sort(key=lambda x: x["relevance_score"], reverse=True)
                    results[filename] = scored_paragraphs[:max_paragraphs]

        logger.info(f"Analysis complete. Found {relevant_paragraphs} relevant paragraphs "
                    f"out of {total_paragraphs} total paragraphs")
        return json.dumps(results)

    except Exception as e:
        logger.error(f"Error analyzing documents: {str(e)}")
        return json.dumps({"error": str(e)})


class TopParagraphsAgent(Agent):
    """Agent that returns highest scoring paragraphs based on relevance."""

    def __init__(self):
        super().__init__(
            name="Top Paragraph Assistant",
            role="Extract most relevant paragraphs from documents",
            model=OpenAILike(
                id=os.getenv("MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("BASE_URL"),
            ),
            tools=[extract_relevant_paragraphs],  # Use the wrapper function directly
            instructions=[
                "Extract the most relevant paragraphs from the specified papers and return raw results"
            ],
            show_tool_calls=True,
            tool_call_limit=1,
            expected_output="extracted document content",
            markdown=True
        )


if __name__ == "__main__":
    try:
        # Example research paper filenames
        test_filenames = [
            "Donington, Jessica (author)_Hu, Xiaohan (author)_Zhang, Su (auth - Real-world Neoadjuvant Treatment Patterns and Outcomes in Resected Non-Small Cell Lung Cancer (2024, Elsevier BV) [10.1016_j.cllc.2024.03.006].pdf"
        ]

        # Initialize the paper analysis agent
        agent = TopParagraphsAgent()

        # Process query
        query = "What are the latest treatments for lung cancer?"
        print(f"\nAnalyzing research papers for query: {query}")

        analysis_response = agent.run(json.dumps(test_filenames))
        print(f"\nPaper analysis results: {analysis_response.content}")

    except Exception as e:
        logger.error(f"Error in paper analysis process: {str(e)}")
