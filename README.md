# AI Oncologist

An intelligent system for analyzing oncological research papers using a multi-agent approach. The system employs three specialized agents to search, extract, and analyze information from medical research papers.

## Architecture

The system consists of three main agents:

1. **Paper Relevance Agent**: Searches and identifies relevant research papers based on the query
2. **Top Paragraphs Agent**: Extracts and ranks the most relevant paragraphs from selected papers
3. **Text Query Agent**: Analyzes the extracted paragraphs to provide focused answers to specific queries

## Prerequisites

- Python 3.8+
- OpenAI API key or compatible API (e.g., DeepSeek)
- PDF files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-oncologist.git
cd ai-oncologist
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```env
# OpenAI-Like API configuration
BASE_URL="https://api.deepseek.com"
OPENAI_API_KEY=your_api_key
MODEL="deepseek-chat"

# Search configuration
MAX_RESULTS=3
SIMILARITY_THRESHOLD=0.3
```

- `BASE_URL`: API endpoint for the LLM service (default: "https://api.deepseek.com")
- `OPENAI_API_KEY`: Your API key for authentication
- `MODEL`: The model to use (default: "deepseek-chat")
- `MAX_RESULTS`: Maximum number of papers to return (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for document selection (default: 0.3)


## Project Structure

```
ai-oncologist/
├── requirements.txt
├── agents/
│   ├── PaperRelevanceAgent.py
│   ├── TextQueryAgent.py
│   └── TopParagraphsAgent.py
├── documents/
└── AIOncologist.py
```

## Usage

1. Place your PDF research papers in the `documents/` directory.

2. Run the main script:
```bash
python AIOncologist.py
```


## Agent Details

### Paper Relevance Agent
- Searches through PDF documents in the documents directory
- Uses embeddings and cosine similarity for initial filtering
- Verifies relevance using LLM-based analysis
- Returns a list of most relevant paper filenames

### Top Paragraphs Agent
- Extracts text from identified papers
- Splits content into manageable chunks
- Scores paragraph relevance using LLM
- Returns top-scoring paragraphs with relevance scores

### Text Query Agent
- Analyzes provided text passages
- Generates focused answers to specific queries
- Uses contextual understanding to provide accurate responses



