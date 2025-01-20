import os
import logging
from typing import Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from phi.model.openai import OpenAILike
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
from phi.agent import Agent

# Configure logging and load environment variables
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
load_dotenv()


def relevant_documents(query: str, return_scores: bool = False) -> str:
    """Return documents relevant to the query."""
    # Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(project_root, "documents")
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    max_results = int(os.getenv("MAX_RESULTS", "10"))

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create docs directory if needed
    if not os.path.exists(docs_path):
        logger.warning(f"Documents directory not found. Creating: {docs_path}")
        os.makedirs(docs_path)

    def load_pdfs() -> Dict[str, str]:
        """Load PDFs from the documents directory."""
        if not os.path.exists(docs_path):
            logger.error(f"Documents directory does not exist: {docs_path}")
            return {}

        documents_by_file = {}
        try:
            pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files")

            for filename in pdf_files:
                try:
                    file_path = os.path.join(docs_path, filename)
                    loader = PyPDFLoader(file_path)
                    pages = loader.load_and_split()
                    full_text = " ".join(page.page_content for page in pages)
                    documents_by_file[filename] = full_text
                    logger.info(f"Loaded {filename} ({len(pages)} pages)")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {str(e)}")

            return documents_by_file
        except Exception as e:
            logger.error(f"Error loading PDFs: {str(e)}")
            return {}

    def verify_relevance(text: str, query: str) -> bool:
        """Verify document relevance using AI verification."""
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
                     "content": "You are a document relevance analyzer. Respond only with 'yes' or 'no'."},
                    {"role": "user",
                     "content": f"Is this document relevant to: '{query}'?\n\nDocument: {text[:1000]}..."}
                ]
            )

            return response.choices[0].message.content.strip().lower() == 'yes'
        except Exception as e:
            logger.error(f"Error during relevance verification: {str(e)}")
            return False

    # Main search logic
    logger.info(f"Searching documents for: {query}")
    try:
        docs_by_file = load_pdfs()
        if not docs_by_file:
            return json.dumps([])

        texts = list(docs_by_file.values())
        filenames = list(docs_by_file.keys())

        # Generate embeddings
        doc_embeddings = embeddings.embed_documents(texts)
        query_embedding = embeddings.embed_query(query)

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Process results
        results = []
        for i, (score, filename) in enumerate(zip(similarities, filenames)):
            if score > similarity_threshold and verify_relevance(texts[i], query):
                if return_scores:
                    results.append({
                        "filename": filename,
                        "relevance_score": round(float(score), 4)
                    })
                else:
                    results.append(filename)

            if len(results) >= max_results:
                break

        return json.dumps(results)

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return json.dumps([])


class PaperRelevanceAgent(Agent):
    """Agent for searching research papers based on queries."""

    def __init__(self):
        super().__init__(
            name="Paper Search Agent",
            role="Search for research papers relevant to the query",
            model=OpenAILike(
                id=os.getenv("MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("BASE_URL"),
            ),
            tools=[relevant_documents],
            instructions=[
                "Search for research papers that are most relevant to the query.",
                "Return a list of filenames"
            ],
            show_tool_calls=True,
            tool_call_limit=1,
            expected_output="list of filenames",
            markdown=True
        )


if __name__ == "__main__":
    try:
        agent = PaperRelevanceAgent()
        query = "What are the latest developments in lung cancer? "
        print(f"\nProcessing research query: {query}")

        search_response = agent.run(query)
        print(f"\nRelevant papers found: {search_response.content}")

    except Exception as e:
        logger.error(f"Error in paper search process: {str(e)}")