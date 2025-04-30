from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Callable, TypeVar, Any, cast, Coroutine
import time
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")


def typed_get(
    path: str, response_model: type[T]
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return cast(
        Callable[[Callable[..., T]], Callable[..., T]],
        app.get(path, response_model=response_model),
    )


def typed_post(
    path: str, response_model: type[T]
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    return cast(
        Callable[
            [Callable[..., Coroutine[Any, Any, T]]],
            Callable[..., Coroutine[Any, Any, T]],
        ],
        app.post(path, response_model=response_model),
    )


def typed_limit(rate: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return cast(Callable[[Callable[..., Any]], Callable[..., Any]], limiter.limit(rate))


def typed_exception_handler(
    exc_class: type[Exception],
) -> Callable[[Callable[..., JSONResponse]], Callable[..., JSONResponse]]:
    return cast(
        Callable[[Callable[..., JSONResponse]], Callable[..., JSONResponse]],
        app.exception_handler(exc_class),
    )


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    average_processing_time: float
    last_error: Optional[str] = None


class Metrics:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.last_error: Optional[str] = None

    def add_request(self, processing_time: float) -> None:
        self.total_requests += 1
        self.total_processing_time += processing_time

    def set_error(self, error: str) -> None:
        self.last_error = error

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_average_processing_time(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time / self.total_requests


app = FastAPI()
metrics = Metrics()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TOKENS = 256


def count_tokens(text: str) -> int:
    tokenizer = model.tokenizer
    return len(tokenizer.encode(text, add_special_tokens=True))


@typed_get("/health", response_model=HealthResponse)
@typed_limit("5/minute")
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        uptime_seconds=metrics.get_uptime(),
        total_requests=metrics.total_requests,
        average_processing_time=metrics.get_average_processing_time(),
        last_error=metrics.last_error,
    )


@typed_post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    start_time = time.time()
    try:
        too_long = [i for i, t in enumerate(req.texts) if count_tokens(t) > MAX_TOKENS]
        if too_long:
            error_msg = f"Text(s) at index(es) {too_long} exceed the max token limit of {MAX_TOKENS}"
            metrics.set_error(error_msg)
            raise HTTPException(
                status_code=422,
                detail=error_msg,
            )

        embs = model.encode(req.texts, show_progress_bar=False)
        processing_time = time.time() - start_time
        metrics.add_request(processing_time)

        return EmbedResponse(embeddings=embs.tolist())
    except Exception as e:
        error_msg = str(e)
        metrics.set_error(error_msg)
        logger.error(f"Error processing request: {error_msg}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {error_msg}"
        )


@typed_exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_msg = str(exc)
    metrics.set_error(error_msg)
    logger.error(f"Unhandled exception: {error_msg}")
    return JSONResponse(
        status_code=500, content={"detail": f"Internal server error: {error_msg}"}
    )
