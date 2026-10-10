"""Mathematical formulations for in-distribution scoring and manifold metrics."""

import math
import re
from typing import Dict

from prompt_manifold.schemas import IntentSlots


class DistributionEvaluator:
    """Evaluates syntactic density, manifold proximity, and slot completeness."""

    def compute_all_metrics(
        self,
        raw_prompt: str,
        enhanced_prompt: str,
        slots: IntentSlots,
        drift_score: float,
    ) -> Dict[str, float]:
        """Compute complete metrics telemetry suite."""
        raw_words = re.findall(r"\b\w+\b", raw_prompt)
        enh_words = re.findall(r"\b\w+\b", enhanced_prompt)

        raw_len = max(1, len(raw_words))
        enh_len = max(1, len(enh_words))

        expansion_ratio = round(enh_len / raw_len, 2)
        slot_coverage = self.compute_slot_coverage(slots)
        in_dist_score = self.compute_in_distribution_score(enh_len, slot_coverage, enh_words)
        quality_index = self.compute_manifold_quality_index(
            in_dist_score, drift_score, slot_coverage
        )

        return {
            "in_distribution_score": in_dist_score,
            "expansion_ratio": max(1.0, expansion_ratio),
            "semantic_drift_score": drift_score,
            "slot_coverage_score": slot_coverage,
            "manifold_quality_index": quality_index,
        }

    def compute_slot_coverage(self, slots: IntentSlots) -> float:
        """Measure the proportion of key syntactic prompt dimensions populated."""
        total_dimensions = 6.0
        active_count = 1.0  # Subject is mandatory
        if slots.environment:
            active_count += 1.0
        if slots.lighting:
            active_count += 1.0
        if slots.camera_framing:
            active_count += 1.0
        if slots.color_palette:
            active_count += 1.0
        if slots.atmosphere:
            active_count += 1.0
        return round(active_count / total_dimensions, 4)

    def compute_in_distribution_score(
        self,
        token_count: int,
        slot_coverage: float,
        tokens: list[str],
    ) -> float:
        """Calculate in-distribution proximity index in [0.0, 1.0].

        Evaluates length bell-curve around empirical foundation model training captions,
        syntactic slot coverage, and lexical vocabulary richness.
        """
        # Bell curve scoring for token length (optimal ~85 words, sigma ~35)
        optimal_len = 85.0
        sigma = 35.0
        length_factor = math.exp(-0.5 * ((token_count - optimal_len) / sigma) ** 2)

        # Token uniqueness / lexical entropy
        unique_ratio = len(set(tokens)) / max(1, len(tokens)) if tokens else 0.5
        lexical_factor = min(1.0, unique_ratio * 1.2)

        # Weighted combination
        score = (0.45 * length_factor) + (0.40 * slot_coverage) + (0.15 * lexical_factor)
        return round(min(1.0, max(0.05, score)), 4)

    def compute_manifold_quality_index(
        self, in_dist_score: float, drift_score: float, slot_coverage: float
    ) -> float:
        """Composite index reflecting overall prompt enhancement efficacy."""
        fidelity_retention = 1.0 - drift_score
        quality = (
            (0.40 * in_dist_score)
            + (0.35 * fidelity_retention)
            + (0.25 * slot_coverage)
        )
        return round(min(1.0, max(0.0, quality)), 4)
