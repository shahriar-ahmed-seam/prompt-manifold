"""Quickstart example demonstrating basic PromptManifold usage."""

from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget


def main():
    engine = PromptManifoldEngine()

    # The canonical infographic case: sparse input
    raw_query = "a teddy bear reading a book"
    print(f"Original Raw Query: '{raw_query}'\n")

    # Project onto Flux manifold
    result = engine.enhance(
        raw_query,
        ExpansionConfig(target_model=ManifoldTarget.FLUX, detail_level=3),
    )

    print("Projected In-Distribution Prompt (Flux):")
    print(result.enhanced_prompt)
    print("\nDual Negative Steering Vector:")
    print(result.negative_prompt)
    print("\nTelemetry Diagnostics:")
    print(f"In-Distribution Score: {result.in_distribution_score:.3f}")
    print(f"Expansion Ratio:       {result.expansion_ratio:.1f}x")
    print(f"Semantic Drift Penalty: {result.semantic_drift_score:.3f}")
    print(f"Latency:               {result.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
