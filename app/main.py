from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MAX_TOKENS = 256


def count_tokens(text: str) -> int:
    # Use the model's tokenizer to count tokens
    tokenizer = model.tokenizer
    return len(tokenizer.encode(text, add_special_tokens=True))


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    too_long = [
        i for i, t in enumerate(req.texts)
        if count_tokens(t) > MAX_TOKENS
    ]
    if too_long:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text(s) at index(es) {too_long} exceed the max token limit"
                f"of {MAX_TOKENS}. Please split or shorten them."
            ),
        )
    embs = model.encode(req.texts, show_progress_bar=False)
    return EmbedResponse(embeddings=embs.tolist())
