"""Streaming client example demonstrating real-time token projection."""

import asyncio
import sys

from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget
from prompt_manifold.streaming.streamer import PromptStreamer


async def main():
    streamer = PromptStreamer()
    prompt = "cyberpunk street at night"

    print(f"Streaming in-distribution tokens for: '{prompt}' (Midjourney manifold):\n")

    cfg = ExpansionConfig(target_model=ManifoldTarget.MIDJOURNEY)
    async for token in streamer.stream_tokens(prompt, cfg, chunk_delay_ms=25.0):
        sys.stdout.write(token)
        sys.stdout.flush()

    print("\n\nStreaming complete.")


if __name__ == "__main__":
    asyncio.run(main())
