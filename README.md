# Fake News Detector
This project is an AI-powered Fake News Detection System built using FastAPI and OpenAI's GPT-4. It analyzes news articles for factual accuracy, bias, and credibility using real-time web search from trusted domains.

## Features
- **Fake News Detection**: Classifies whether an article is real, fake, or partially true using evidence from credible sources.
- **Fact Checking**: Extracts key claims and verifies them against trustworthy domains.
- **Bias Analysis**: Identifies political leanings, emotional language, and biased framing.
- **Credibility Score**: Provides an overall credibility score for any news article.

## How It Works
1. **Article Input**: The user submits a news article.
2. **Keyword Extraction**: The system extracts key factual claims from the article.
3. **Domain Selection**: It selects relevant trusted websites based on topic and location.
4. **Web Search**: Searches for each claim using the Tavily search tool.
5. **Analysis**: Compares claims against sources and rates them.
6. **Judgment**: Returns a credibility score, justification, and source links.

## Tech Stack
- FastAPI
- LangChain + OpenAI GPT-4
- Tavily Search API
- Python
- Jinja2 Templates
- HTML/CSS (for frontend)

## Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```
### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Add a .env file
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the application
```bash
uvicorn detector2:app --reload
```
## Project Structure
```bash
fake-news-detector/
├── main.py                 # FastAPI backend logic
├── templates/
│   └── index.html          # Frontend HTML template
├── static/                 # CSS/JS/Assets
├── logs.csv                # Analysis logs
├── .env                    # Environment secrets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 📡 API Endpoints

### `POST /detect`
Analyzes the article and gives a **real/fake judgment** with score and sources.

#### Request:
```json
{
  "article": "Your news article text here"
}
```

#### Response:
```json
{
  "result": {
    "score": 74,
    "justification": "Claim 1 is supported... Claim 2 is false...",
    "sources": [
      "https://reuters.com/...",
      "https://who.int/..."
    ]
  }
}

```

### 'POST /fact-check'
Returns a detailed fact-checking report in Markdown format.
#### Request:
```json
{
  "article": "Your article here"
}

```
#### Response:
```json
{
  "result": "Markdown report with evidence, ratings, and source analysis."
}

```

### POST /bias-analysis

#### Request:
```json
{
  "article": "Your article here"
}

```
#### Response:
```json
{
  "result": "Markdown report highlighting biased phrases and political lean."
}

```

