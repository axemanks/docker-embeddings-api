from fastapi.testclient import TestClient
from app.main import app, MAX_TOKENS

client = TestClient(app)


# Helper to generate text with a given number of tokens (approximate)
def gen_text(token_count):
    return "word " * token_count


# Test valid short text
def test_embed_valid_short_text():
    resp = client.post("/embed", json={"texts": ["hello world!"]})
    assert resp.status_code == 200
    assert "embeddings" in resp.json()
    assert len(resp.json()["embeddings"]) == 1


# Test text exceeds token limit
def test_embed_text_exceeds_token_limit():
    long_text = gen_text(MAX_TOKENS + 1)
    resp = client.post("/embed", json={"texts": [long_text]})
    assert resp.status_code == 422
    assert "exceed the max token limit" in resp.text


# Test batch mixed valid and invalid
def test_embed_batch_mixed_valid_invalid():
    valid = "short text"
    invalid = gen_text(MAX_TOKENS + 5)
    resp = client.post("/embed", json={"texts": [valid, invalid, valid]})
    assert resp.status_code == 422
    assert "exceed the max token limit" in resp.text


# Test large batch
def test_embed_large_batch():
    texts = ["test"] * 100  # Simulate a job with 100 short texts
    resp = client.post("/embed", json={"texts": texts})
    assert resp.status_code == 200
    assert len(resp.json()["embeddings"]) == 100
