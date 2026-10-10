"""High-throughput batch processing example with PromptManifold."""

from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget


def main():
    engine = PromptManifoldEngine()

    sparse_prompts = [
        "a teddy bear reading a book",
        "meat",
        "cyberpunk street at night",
        "product photo running shoes",
        "portrait of an elderly watchmaker",
    ]

    print(f"Executing batch projection on {len(sparse_prompts)} sparse queries...\n")
    results = engine.enhance_batch(
        sparse_prompts,
        ExpansionConfig(target_model=ManifoldTarget.SDXL),
    )

    for i, res in enumerate(results, 1):
        print(f"[{i}] RAW: '{res.raw_input}'")
        print(f"    ENHANCED (SDXL): {res.enhanced_prompt[:85]}...")
        print(
            f"    S_ID: {res.in_distribution_score:.3f} | "
            f"Expansion: {res.expansion_ratio:.1f}x | "
            f"Latency: {res.latency_ms:.2f}ms\n"
        )


if __name__ == "__main__":
    main()
