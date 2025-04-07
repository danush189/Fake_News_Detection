from fastapi import FastAPI
from pydantic import BaseModel
from detector4 import detect_fake_news

app = FastAPI()

class ArticleRequest(BaseModel):
    article: str

@app.post("/detect")
async def detect(article_request: ArticleRequest):
    result = await detect_fake_news(article_request.article)
    return {"result": result}