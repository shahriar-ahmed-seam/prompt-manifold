# Changelog

All notable changes to PromptManifold will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

### Added
- Core `PromptManifoldEngine` for deterministic inference-time prompt distribution alignment.
- Syntactic `IntentExtractor` isolating subject anchors, lighting schemas, and camera optics.
- Foundation model adapters for Flux, SDXL, Stable Diffusion 3.5, Midjourney, and DALL-E 3.
- Dual manifold negative prompt generation with anti-artifact constraints.
- `SemanticDriftGuard` enforcing entity retention and measuring drift penalties ($\Delta_{\text{drift}}$).
- `DistributionEvaluator` calculating in-distribution proximity index ($S_{\text{ID}}$) and slot coverage ($C_{\text{slots}}$).
- High-throughput asynchronous `PromptStreamer` emitting Server-Sent Events (SSE).
- Production FastAPI service exposing `/v1/enhance`, `/v1/enhance/stream`, `/v1/benchmark`, and `/v1/health`.
- Typer and Rich operational CLI (`manifold enhance`, `manifold benchmark`, `manifold inspect`, `manifold serve`).
- Anti-AI-slop developer telemetry console (`src/prompt_manifold/ui/index.html`).
- Empirical benchmark suite with 100% win rate across canonical scenarios.
- Authoritative documentation blueprints (`docs/ARCHITECTURE.md`, `docs/SPECIFICATION.md`, `docs/BENCHMARKS.md`).
