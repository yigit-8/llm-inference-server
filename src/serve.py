"""
LLM inference API backed by the continuous batching engine.

Requests are handed to the engine thread and the handler blocks until the engine
finishes them. Handlers are plain `def`, so FastAPI runs them in its thread pool
and a blocking wait costs a pool thread rather than the event loop.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import metrics
from src.config import settings
from src.runner import EngineRunner

runner: EngineRunner | None = None


def load_runner() -> EngineRunner:
    logger.info(f"loading {settings.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME)
    model.eval()
    return EngineRunner(model, tokenizer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = load_runner()
    runner.start()
    yield
    runner.stop()


app = FastAPI(
    title="LLM Inference Server",
    description="Continuous batching over a causal LM, with TTFT and throughput metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.get("/metrics", include_in_schema=False)(metrics.metrics_endpoint)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=settings.MAX_NEW_TOKENS, ge=1, le=512)


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    time_to_first_token_seconds: float | None
    total_seconds: float


@app.get("/health")
def health():
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine not started.")
    if not runner.is_healthy():
        # The process is up but the engine thread is not, so nothing can be
        # generated. Reporting "ok" here would keep a broken pod in the rotation.
        raise HTTPException(
            status_code=503,
            detail=f"Engine thread is not running: {runner.error_message or 'unknown error'}",
        )
    return {"status": "ok", "model": settings.MODEL_NAME}


@app.get("/stats")
def stats():
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine not started.")
    return runner.stats()


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if runner is None:
        raise HTTPException(status_code=503, detail="Engine not started.")

    sequence, event = runner.submit(request.prompt, request.max_new_tokens)
    if not runner.wait(sequence, event, settings.REQUEST_TIMEOUT_SECONDS):
        raise HTTPException(status_code=504, detail="Generation timed out.")

    total = (sequence.finished_at or 0.0) - sequence.queued_at
    return GenerateResponse(
        text=runner.tokenizer.decode(sequence.tokens, skip_special_tokens=True),
        tokens=len(sequence.tokens),
        time_to_first_token_seconds=sequence.time_to_first_token(),
        total_seconds=total,
    )
