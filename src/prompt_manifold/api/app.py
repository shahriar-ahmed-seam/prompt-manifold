"""FastAPI production service for PromptManifold."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from prompt_manifold import __version__
from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.eval.benchmark import BenchmarkRunner
from prompt_manifold.schemas import EnhancedPromptResult, ExpansionConfig, ManifoldTarget
from prompt_manifold.streaming.streamer import PromptStreamer


class EnhanceRequest(BaseModel):
    """Enhancement request payload."""
    prompt: str = Field(..., description="Sparse user prompt to project onto manifold.")
    target: ManifoldTarget = Field(
        default=ManifoldTarget.FLUX, description="Target foundation model."
    )
    detail_level: int = Field(default=3, ge=1, le=5, description="Detail level 1-5.")
    generate_negative: bool = Field(default=True, description="Generate negative prompt.")


app = FastAPI(
    title="PromptManifold API",
    description="Inference-Time Prompt Distribution Alignment & Manifold Projection Engine",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PromptManifoldEngine()
streamer = PromptStreamer(engine)
benchmark_runner = BenchmarkRunner(engine)


@app.get("/v1/health")
def health_check():
    """Service health and runtime information."""
    return {
        "status": "healthy",
        "service": "prompt-manifold",
        "version": __version__,
        "engine": "PromptManifoldEngine-v1",
    }


@app.post("/v1/enhance", response_model=EnhancedPromptResult)
def enhance_prompt(req: EnhanceRequest):
    """Project sparse prompt synchronously onto target model manifold."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    config = ExpansionConfig(
        target_model=req.target,
        detail_level=req.detail_level,
        generate_negative=req.generate_negative,
    )
    return engine.enhance(req.prompt, config)


@app.post("/v1/enhance/stream")
async def stream_enhance_prompt(req: EnhanceRequest):
    """Stream token projections via Server-Sent Events (SSE)."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    config = ExpansionConfig(
        target_model=req.target,
        detail_level=req.detail_level,
        generate_negative=req.generate_negative,
    )

    async def event_generator():
        async for ev in streamer.stream_events(req.prompt, config, chunk_delay_ms=10.0):
            yield f"event: {ev['event']}\ndata: {ev['data']}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/benchmark")
def run_benchmark():
    """Execute benchmark suite and return comparative telemetry."""
    return benchmark_runner.run_suite()


@app.get("/", response_class=HTMLResponse)
def serve_console():
    """Serve the developer telemetry console."""
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>PromptManifold API Online</h1>")
