import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
search_tool = TavilySearchResults()

# FastAPI app
app = FastAPI(title="Fake News Detector")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Config ---
TRUSTED_DOMAINS = {
    "health": ["who.int", "cdc.gov", "medicalnewstoday.com"],
    "politics": ["reuters.com", "apnews.com", "bbc.com"],
    "science": ["nationalgeographic.com", "sciencenews.org"],
    "factcheck": ["snopes.com", "factcheck.org"],
    "general": ["bbc.com", "reuters.com", "apnews.com", "npr.org", "theguardian.com"]
}

LOCATION_DOMAINS = {
    "india": ["thehindu.com", "indiatoday.in", "timesofindia.indiatimes.com"],
    "south india": ["thehindu.com", "deccanherald.com"],
    "north india": ["hindustantimes.com", "ndtv.com"],
    "central india": ["freepressjournal.in", "patrika.com"],
    "west india": ["dnaindia.com", "mumbaimirror.indiatimes.com"],
    "eastern india": ["telegraphindia.com"],
    "usa": ["nytimes.com", "cnn.com", "washingtonpost.com"],
    "default": []
}

LOG_FILE = "logs.csv"

class ArticleInput(BaseModel):
    article: str


# --- Extractor ---
def extract_keywords(article: str):
    prompt = f"""
    You are 'The Extractor'. Your job is to extract 5 to 10 factual claims or key phrases from the following article that need fact-checking.
    These should be short, precise, and suitable to be used as search queries.

    Return a Python list of strings. If the article has no valid facts to check, return an empty list: []

    Article:
    {article}
    """
    response = llm.predict(prompt).strip()
    try:
        keywords = eval(response)
        if not isinstance(keywords, list):
            raise ValueError("Not a list")
        return keywords
    except Exception as e:
        print("❌ Keyword extraction failed:", e)
        print("LLM returned:", response)
        return []


# --- Strategist ---
def choose_domains(keywords, article):
    domain_map = []
    for keyword in keywords:
        prompt = f"""
        You are 'The Strategist'. Based on the keyword '{keyword}' and the article context below, decide:
        1. The content category (choose from: health, politics, science, factcheck, general)
        2. The most likely region or country the news relates to (e.g., india, usa, south india, central india, etc.)

        Article:
        {article}

        Return format as Python tuple: ("<category>", "<region>")
        """
        response = llm.predict(prompt).strip().lower()
        try:
            category, region = eval(response)
        except:
            category, region = "general", "default"

        category_domains = TRUSTED_DOMAINS.get(category, TRUSTED_DOMAINS["general"])
        location_domains = LOCATION_DOMAINS.get(region, [])

        if category != "general" and region == "default":
            domains = category_domains[:2]
        elif region != "default" and category == "general":
            domains = location_domains[:2]
        elif category != "general" and region != "default":
            domains = category_domains[:2] + location_domains[:2]
        else:
            domains = category_domains[:1] + location_domains[:1]

        domain_map.append((keyword, list(dict.fromkeys(domains))))
    return domain_map


# --- Scout ---
def search_tavily(keyword: str, domains: list):
    joined_domains = " OR ".join([f"site:{d}" for d in domains])
    search_query = f"{keyword} {joined_domains}"
    print(f"\n🔍 Searching for: {search_query}")
    result = search_tool.run(search_query)
    return result


# --- Fact Checker ---
def analyze_factuality(article, keywords, search_data):
    prompt = f"""
You are 'The Fact Checker', a specialist in verifying claims from news articles.

📰 Article:
{article}

🔑 Key Claims:
{keywords}

🔍 Search Results:
{search_data}

Your job:
- Compare each claim in the article with the search results from trusted sources
- Assess the factual accuracy of each claim
- Rate each claim as TRUE, FALSE, MISLEADING, or UNVERIFIABLE
- Provide source evidence for your conclusions

Return a structured report in markdown with:
1. Overall factuality score (0-100%)
2. Analysis of each key claim with evidence
3. List of verified facts vs. misinformation
4. Summary of factual accuracy
"""
    return llm.predict(prompt)


# --- Bias Detector ---
def analyze_bias(article, keywords):
    prompt = f"""
You are 'The Bias Detector', an expert in media literacy and bias identification.

📰 Article:
{article}

🔑 Key Topics:
{keywords}

Your job:
- Analyze the article for indicators of bias
- Identify emotional language, framing techniques, and partisan terms
- Assess whether the article shows political leaning (left, right, or center)
- Detect loaded language, opinion statements presented as facts, and narrative techniques

Return a structured report in markdown with:
1. Overall bias score (0-100%, where 0 is perfectly neutral and 100 is extremely biased)
2. Political leaning assessment
3. Examples of biased language or framing from the text
4. Analysis of tone and emotion in the writing
5. Assessment of how the article presents opposing viewpoints
"""
    return llm.predict(prompt)


# --- Judge ---
def judge_realness(article, keywords, search_data):
    prompt = f"""
You are 'The Judge', a top-tier AI fake news analyst. You've been given:

📰 Full Article:
{article}

🔑 Extracted Keywords:
{keywords}

🔍 Search Results from trusted sources:
{search_data}

Your job:
- Compare claims in the article with facts found in the search results.
- Identify whether the article is fake or real.
- Consider both content domain and geographic location.
- Give a clear justification explaining any partial truth, contradictions, or factual alignment.

Return a well-formatted JSON with:
{{
  "score": <integer from 0 (completely real) to 100 (completely fake)>,
  "justification": "<detailed reasoning including which claims were verified or contradicted>",
  "sources": ["<link1>", "<link2>", ...]
}}
"""
    response = llm.predict(prompt).strip()
    try:
        return eval(response)
    except Exception as e:
        print("❌ Failed to parse judge response:", e)
        return {
            "score": 0,
            "justification": "Failed to parse LLM output",
            "sources": []
        }


# --- Logger ---
def log_to_csv(article, keywords, domain_map, search_results, judgment):
    headers = [
        "timestamp", "article", "keywords", "domain_map",
        "search_results", "score", "justification", "sources"
    ]

    write_headers = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if write_headers:
            writer.writeheader()

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "article": article,
            "keywords": str(keywords),
            "domain_map": str(domain_map),
            "search_results": str(search_results),
            "score": judgment.get("score", ""),
            "justification": judgment.get("justification", ""),
            "sources": str(judgment.get("sources", [])),
        }

        writer.writerow(row)



# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect_fake_news(payload: ArticleInput):
    article = payload.article
    keywords = extract_keywords(article)
    if not keywords:
        return {"error": "No keywords found"}

    domain_map = choose_domains(keywords, article)
    search_results = {k: search_tavily(k, d) for k, d in domain_map}
    judgment = judge_realness(article, keywords, search_results)

    log_to_csv(article, keywords, domain_map, search_results, judgment)
    return {"result": judgment}


@app.post("/fact-check")
async def fact_check(payload: ArticleInput):
    article = payload.article
    keywords = extract_keywords(article)
    if not keywords:
        return {"error": "No keywords found"}

    domain_map = choose_domains(keywords, article)
    search_results = {k: search_tavily(k, d) for k, d in domain_map}
    return {"result": analyze_factuality(article, keywords, search_results)}


@app.post("/bias-analysis")
async def bias_analyze(payload: ArticleInput):
    article = payload.article
    keywords = extract_keywords(article)
    if not keywords:
        return {"error": "No keywords found"}
    return {"result": analyze_bias(article, keywords)}


@app.post("/credibility-score")
async def credibility_score(payload: ArticleInput):
    article = payload.article
    keywords = extract_keywords(article)
    if not keywords:
        return {"error": "No keywords found"}

    domain_map = choose_domains(keywords, article)
    search_results = {k: search_tavily(k, d) for k, d in domain_map}
    judgment = judge_realness(article, keywords, search_results)

    return judgment


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
