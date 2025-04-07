from pydantic import BaseModel

class ArticleInput(BaseModel):
    content: str
