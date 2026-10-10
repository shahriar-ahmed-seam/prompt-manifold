"""Streaming token generation and Server-Sent Events (SSE) for PromptManifold."""

import asyncio
import json
from typing import AsyncIterator, Dict, Optional

from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.schemas import EnhancedPromptResult, ExpansionConfig


class PromptStreamer:
    """Streams enhanced prompt projection chunks for real-time inference pipelines."""

    def __init__(self, engine: Optional[PromptManifoldEngine] = None):
        self.engine = engine or PromptManifoldEngine()

    async def stream_tokens(
        self,
        raw_prompt: str,
        config: Optional[ExpansionConfig] = None,
        chunk_delay_ms: float = 8.0,
    ) -> AsyncIterator[str]:
        """Stream projected prompt token-by-token."""
        result = self.engine.enhance(raw_prompt, config)
        words = result.enhanced_prompt.split(" ")

        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            if chunk_delay_ms > 0:
                await asyncio.sleep(chunk_delay_ms / 1000.0)

    async def stream_events(
        self,
        raw_prompt: str,
        config: Optional[ExpansionConfig] = None,
        chunk_delay_ms: float = 8.0,
    ) -> AsyncIterator[Dict[str, str]]:
        """Stream SSE-compatible events including lifecycle metadata and final telemetry."""
        cfg = config or ExpansionConfig()
        yield {
            "event": "start",
            "data": json.dumps({"raw_prompt": raw_prompt, "target": cfg.target_model.value}),
        }

        result: EnhancedPromptResult = self.engine.enhance(raw_prompt, cfg)
        words = result.enhanced_prompt.split(" ")

        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield {
                "event": "token",
                "data": json.dumps({"token": token, "index": i}),
            }
            if chunk_delay_ms > 0:
                await asyncio.sleep(chunk_delay_ms / 1000.0)

        yield {
            "event": "complete",
            "data": json.dumps({
                "enhanced_prompt": result.enhanced_prompt,
                "negative_prompt": result.negative_prompt,
                "in_distribution_score": result.in_distribution_score,
                "expansion_ratio": result.expansion_ratio,
                "semantic_drift_score": result.semantic_drift_score,
                "slot_coverage_score": result.slot_coverage_score,
                "latency_ms": result.latency_ms,
            }),
        }
