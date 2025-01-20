from dotenv import load_dotenv
from phi.agent import Agent
import os
import logging

from phi.model.openai import OpenAILike

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
load_dotenv()

class TextQueryAnswerAgent(Agent):
    """Agent that answers queries by analyzing provided text passages."""
    def __init__(self):
        super().__init__(
            name="Text Query Answer Assistant",
            role="Answer questions by analyzing provided text passages",
            model=OpenAILike(
                id=os.getenv("MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("BASE_URL"),
            ),
            instructions=[
                "Answer queries using only the provided text passages",
                "Base answers strictly on the given text content",
                "Provide clear answers focused on the query"
            ],
            show_tool_calls=True,
            markdown=True
        )

if __name__ == "__main__":
    try:
        # Initialize the agent
        answer_agent = TextQueryAnswerAgent()

        # Test data
        test_query = "What are the latest treatments for lung cancer?"
        test_paragraphs = """1. Canakinumab in Combination with Docetaxel (Paz-Ares et al., 2024)  
            Relevance Score: 0.9
            Key Findings:
            - The study investigates the efficacy of canakinumab (an IL-1β inhibitor) in combination with docetaxel for treating advanced Non-Small Cell Lung Cancer (NSCLC).
            - Patients included had progressed after platinum-based doublet chemotherapy (PDC) and immunotherapy.
            - The combination aims to improve survival outcomes compared to docetaxel alone.
            - This is a phase 3 trial (CANOPY-2), highlighting its significance in clinical practice.
            
            2. Chemo-Immunotherapy Combinations (Gridelli et al., 2024)  
            Relevance Score: 1.0
            Key Findings:
            - The integration of immune checkpoint inhibitors (ICIs) with platinum-based chemotherapy is now a standard of care for advanced NSCLC.
            - Trials like IMpower130, CheckMate 9LA, and POSEIDON demonstrate improved efficacy of chemo-immunotherapy combinations over chemotherapy alone.
            - Specific combinations include:
               • Atezolizumab + carboplatin + nab-paclitaxel (IMpower130)
               • Nivolumab + ipilimumab + chemotherapy (CheckMate 9LA)
               • Durvalumab ± tremelimumab + chemotherapy (POSEIDON)
            - These combinations have shown prolonged survival and are approved by regulatory agencies like the FDA and EMA.
            
            3. Emerging Trends in NSCLC Treatment  
            Relevance Score: 0.9
            Key Findings:
            - The combination of chemotherapy and immunotherapy represents a breakthrough in treating NSCLC, especially for patients without EGFR/ALK alterations.
            - Challenges remain in selecting the optimal regimen for individual patients, given the variety of available options."""

        # Create input combining query and paragraphs
        input_text = f"""Query: {test_query}. Relevant Research Paragraphs: {test_paragraphs}"""

        # Generate answer
        logger.info("Generating answer from query and paragraphs...")
        answer_agent.print_response(input_text, stream=True)

    except Exception as e:
        logger.error(f"Error in text query answer process: {str(e)}")