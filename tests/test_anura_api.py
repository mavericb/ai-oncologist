import os
import time
import logging
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI

from agents.PaperRelevanceAgent import PaperRelevanceAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables (optional, hardcoded for now)
load_dotenv()

# API details from Anura example
API_KEY = os.getenv("ANURA_API_KEY")
BASE_URL = os.getenv("ANURA_BASE_URL")
MODEL = os.getenv("ANURA_MODEL")

# Headers for API calls (match curl exactly)
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "text/event-stream"
}

# Test 1: Curl-like API call (2+2)
def test_curl():
    url = f"{BASE_URL}/api/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "you are a helpful AI assistant"},
            {"role": "user", "content": "2+2"}
        ],
        "stream": False,
        "options": {"temperature": 1.0}
    }

    logger.info("Starting curl API test (2+2)...")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=120)
        response.raise_for_status()

        full_content = ""
        for line in response.iter_lines(chunk_size=1):
            if line:
                decoded_line = line.decode('utf-8').strip()
                logger.info(decoded_line)
                if decoded_line.startswith("data: "):
                    chunk = decoded_line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        if "message" in data and "content" in data["message"]:
                            full_content = data["message"]["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {chunk}")
                elif decoded_line.startswith("event: delta"):
                    pass

        elapsed_time = time.time() - start_time
        logger.info(f"Curl-style API worked! Full response: '{full_content}'")
        logger.info(f"Time taken: {elapsed_time:.3f} seconds")
        if "4" in full_content:
            logger.info("Response validated: Contains '4' as expected")
        else:
            logger.warning(f"Unexpected response: Expected '4', got '{full_content}'")
    except requests.RequestException as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Curl-style API failed: {str(e)} (Time taken: {elapsed_time:.3f} seconds)")

# Test 2: OpenAI-compatible API call (2+3) using OpenAI client (expected to fail)
def test_openai_compatible():
    logger.info("Starting OpenAI-compatible API test (2+3)...")
    start_time = time.time()

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=f"{BASE_URL}/api/v1/",
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "you are a helpful AI assistant"},
                {"role": "user", "content": "2+3"}
            ],
            temperature=1.0
        )

        content = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        logger.info(f"OpenAI-compatible API worked! Response: '{content}'")
        logger.info(f"Time taken: {elapsed_time:.3f} seconds")
        if "5" in content:
            logger.info("Response validated: Contains '5' as expected")
        else:
            logger.warning(f"Unexpected response: Expected '5', got '{content}'")
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"OpenAI-compatible API failed: {str(e)} (Time taken: {elapsed_time:.3f} seconds)")
        if 'response' in locals():
            logger.error(f"Response: {response}")

# Verify relevance function (standalone as per your shared code)
def verify_relevance(text: str, query: str) -> bool:
    """Verify document relevance using AI verification."""
    url = f"{BASE_URL}/api/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a document relevance analyzer. Respond only with 'yes' or 'no'."},
            {"role": "user", "content": f"Is this document relevant to: '{query}'?\n\nDocument: {text[:1000]}..."}
        ],
        "stream": False,
        "options": {"temperature": 1.0}
    }

    try:
        response = requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=120)
        response.raise_for_status()

        full_content = ""
        for line in response.iter_lines(chunk_size=1):
            if line:
                decoded_line = line.decode('utf-8').strip()
                logger.info(decoded_line)  # Print each line live like curl
                if decoded_line.startswith("data: "):
                    chunk = decoded_line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        if chunk.startswith("{"):
                            data = json.loads(chunk)
                            if "message" in data and "content" in data["message"]:
                                full_content += data["message"]["content"]
                        else:
                            for sub_chunk in chunk.split('}{'):
                                if not sub_chunk.startswith('{'):
                                    sub_chunk = '{' + sub_chunk
                                if not sub_chunk.endswith('}'):
                                    sub_chunk += '}'
                                data = json.loads(sub_chunk)
                                if "message" in data and "content" in data["message"]:
                                    full_content += data["message"]["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {chunk}")

        cleaned_content = full_content.strip().lower().rstrip('.')
        return cleaned_content == 'yes'
    except requests.RequestException as e:
        logger.error(f"Error during relevance verification: {str(e)}")
        return False

# Test 3: Calls verify_relevance function
def test_verify_relevance():
    logger.info("Starting verify relevance test...")
    start_time = time.time()

    # Test data
    text = """1. Canakinumab in Combination with Docetaxel (Paz-Ares et al., 2024)  
        Relevance Score: 0.9
        Key Findings:
        - The study investigates the efficacy of canakinumab (an IL-1β inhibitor) in combination with docetaxel for treating advanced Non-Small Cell Lung Cancer (NSCLC).
        - Patients included had progressed after platinum-based doublet chemotherapy (PDC) and immunotherapy."""
    query = "lung cancer treatments"

    try:
        is_relevant = verify_relevance(text, query)
        elapsed_time = time.time() - start_time
        logger.info(f"Verify relevance worked! Is relevant: {is_relevant}")
        logger.info(f"Time taken: {elapsed_time:.3f} seconds")
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Verify relevance test failed: {str(e)} (Time taken: {elapsed_time:.3f} seconds)")

# Test 4: Paper relevance agent test (using existing PaperRelevanceAgent)
def test_paper_relevance_agent():
    logger.info("Starting Test 4: Paper Relevance Agent")
    try:
        agent = PaperRelevanceAgent()
        query = "What are the latest developments in lung cancer? "
        logger.info(f"Processing research query: {query}")

        search_response = agent.run(query)
        logger.info(f"Relevant papers found: {search_response.content}")

    except Exception as e:
        logger.error(f"Error in paper search process: {str(e)}")

if __name__ == "__main__":
    test_curl()
    print("\n" + "=" * 50 + "\n")
    test_openai_compatible()
    print("\n" + "=" * 50 + "\n")
    test_verify_relevance()
    print("\n" + "=" * 50 + "\n")
    test_paper_relevance_agent()