import logging
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
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


def extract_relevant_paragraphs(query: str, filenames: Optional[List[str]] = None) -> str:
    """Extract relevant paragraphs."""
    # Configuration and initialization
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(project_root, "documents")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    max_paragraphs = int(os.getenv("MAX_PARAGRAPHS", "5"))

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

    def score_paragraph(paragraph: str, query: str) -> float:
        """Score a paragraph's relevance to the query using OpenAI-Like API."""
        logger.debug(f"Scoring paragraph for query: '{query}'")
        try:
            # TODO Update with lilypad endpoint
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "You are a document analyzer. Given a paragraph and query, analyze their relevance. Respond only with a score from 0.0 to 1.0."},
                    {"role": "user",
                     "content": f"Task: Score how relevant this paragraph is to the query. Score from 0.0 (irrelevant) to 1.0 (highly relevant).\n\nQuery: {query}\nParagraph: {paragraph}\n\nRespond only with the numerical score:"}
                ],
                stream=False,
                temperature=0.1
            )

            score = float(response.choices[0].message.content.strip())
            score = min(max(score, 0.0), 1.0)
            logger.debug(f"Received relevance score: {score:.2f}")
            return score

        except Exception as e:
            logger.error(f"Error scoring paragraph: {str(e)}")
            return 0.0

    # Main analysis logic
    try:
        if filenames is None:
            filenames = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]

        if not filenames:
            logger.warning("No PDF files found to analyze")

        results = {}
        total_paragraphs = 0
        relevant_paragraphs = 0

        for filename in filenames:
            file_path = os.path.join(docs_path, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {filename}")
                continue

            paragraphs = load_document(file_path)
            if not paragraphs:
                continue

            scored_paragraphs = []
            for paragraph in paragraphs:
                total_paragraphs += 1
                score = score_paragraph(paragraph, query)

                if score > relevance_threshold:
                    scored_paragraphs.append({
                        "text": paragraph,
                        "relevance_score": round(score, 4)
                    })
                    relevant_paragraphs += 1

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
            tools=[extract_relevant_paragraphs],
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
            "Donington, Jessica (author)_Hu, Xiaohan (author)_Zhang, Su (auth - Real-world Neoadjuvant Treatment Patterns and Outcomes in Resected Non-Small Cell Lung Cancer (2024, Elsevier BV) [10.1016_j.cllc.2024.03.006].pdf"]

        # Initialize the paper analysis agent
        agent = TopParagraphsAgent()

        # Process query
        query = "What are the latest treatments for lung cancer?"
        print(f"\nAnalyzing research papers for query: {query}")

        analysis_response = agent.run(json.dumps(test_filenames))
        print(f"\nPaper analysis results: {analysis_response.content}")

    except Exception as e:
        logger.error(f"Error in paper analysis process: {str(e)}")