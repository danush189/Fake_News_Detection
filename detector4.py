import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Agents (Personas)
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
search_tool = TavilySearchResults()

app = FastAPI()

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

# --- Request Schema ---
class ArticleInput(BaseModel):
    article: str

# --- The Extractor ---
def extract_keywords(article: str):
    prompt = f"""
    You are 'The Extractor'. Your job is to extract 5 to 10 factual claims or key phrases from the following article that need fact-checking.
    These should be short, precise, and suitable to be used as search queries.

    Article:
    """
    article = article.strip()
    response = llm.predict(prompt + article)
    return eval(response.strip())

# --- The Strategist ---
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

        # New adjusted strategy
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

# --- The Scout ---
def search_tavily(keyword: str, domains: list):
    joined_domains = " OR ".join([f"site:{d}" for d in domains])
    search_query = f"{keyword} {joined_domains}"
    print(f"\n🔍 Searching for: {search_query}")
    result = search_tool.run(search_query)
    return result

# --- The Judge ---
def judge_realness(article, keywords, search_data):
    prompt = f"""
You are 'The Judge', a top-tier AI fake news analyst. You’ve been given:

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
{
  "score": <integer from 0 (completely real) to 100 (completely fake)>,
  "justification": "<detailed reasoning including which claims were verified or contradicted>",
  "sources": ["<link1>", "<link2>", ...]
}
Be honest, analytical, and concise. Quote or rephrase facts where necessary.
"""
    response = llm.predict(prompt)
    return response

# --- FastAPI Route ---
@app.post("/detect")
async def detect_fake_news(payload: ArticleInput):
    article = payload.article

    # Step 1: The Extractor
    keywords = extract_keywords(article)

    # Step 2: The Strategist
    domain_map = choose_domains(keywords, article)

    # Step 3: The Scout
    search_results = {}
    for keyword, domains in domain_map:
        result = search_tavily(keyword, domains)
        search_results[keyword] = result

    # Step 4: The Judge
    judgment = judge_realness(article, keywords, search_results)

    return {"result": judgment}
