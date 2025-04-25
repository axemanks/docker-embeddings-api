# Embedding Service

A FastAPI service for generating sentence embeddings using Hugging Face Sentence Transformers.

## Usage

### Build Docker Image
```sh
docker build -t embedding-service .
```

### Run Container
```sh
docker run -p 8000:8000 embedding-service
```

### Example Request
```json
POST /embed
{
  "texts": ["Hello world", "How are you?"]
}
```

### Response
```json
{
  "embeddings": [[...], [...]]
}
```

## Publishing
To publish to Docker Hub:
1. Login: `docker login`
2. Tag: `docker tag embedding-service <your-dockerhub-username>/embedding-service:latest`
3. Push: `docker push <your-dockerhub-username>/embedding-service:latest`

---

This project is ready for local use, Dockerization, and sharing!
