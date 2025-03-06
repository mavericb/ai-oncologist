import logging
from agents.PaperRelevanceAgent import PaperRelevanceAgent
from agents.TextQueryAgent import TextQueryAnswerAgent
from agents.TopParagraphsAgent import TopParagraphsAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize the agents
        relevance_agent = PaperRelevanceAgent()
        paragraph_agent = TopParagraphsAgent()
        answer_agent = TextQueryAnswerAgent()

        # Example query
        query = "What are the latest developments in lung cancer? "
        print(f"\nProcessing research query: {query}")

        # Run the pipeline
        paper_results = relevance_agent.run(query)
        print(f"\nRelevant papers found: {paper_results.content}")

        paragraph_results = paragraph_agent.run(f"query: {query}, paper_results:{paper_results}")
        print(f"\nRelevant paragraphs found: {paragraph_results.content}")

        final_answer = answer_agent.run(f"query: {query}, paragraph_results:{paragraph_results}")
        print(f"\nFinal answer: {final_answer.content}")

        return final_answer

    except Exception as e:
        logger.error(f"Error in paper search process: {str(e)}")
        raise


if __name__ == "__main__":
    main()