"""Midjourney v6 foundation model manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class MidjourneyAdapter(BaseManifoldAdapter):
    """Adapter optimized for Midjourney v6+ artistic parameters and styling."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.MIDJOURNEY

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        clauses = [slots.subject]

        if slots.subject_attributes:
            clauses.append(", ".join(slots.subject_attributes[:3]))

        if slots.environment:
            clauses.append(slots.environment)

        if slots.lighting:
            clauses.append(slots.lighting)

        if slots.camera_framing:
            clauses.append(slots.camera_framing)

        if slots.color_palette:
            clauses.append(slots.color_palette)

        core_prompt = ", ".join(clauses)

        # Build Midjourney CLI flag parameters
        params = ["--v 6.1", "--style raw", "--ar 16:9"]
        if config.detail_level >= 4:
            params.append("--stylize 300")
        else:
            params.append("--stylize 150")

        full_prompt = f"{core_prompt} {' '.join(params)}"
        metadata = {
            "target": self.target.value,
            "engine": "midjourney_v6",
            "cli_flags": params,
        }
        return full_prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        # Midjourney uses --no flag internally
        base_neg = ["blurry", "distorted", "text", "watermark"]
        if slots.explicit_negatives:
            base_neg = slots.explicit_negatives + base_neg
        return " --no " + " ".join(dict.fromkeys(base_neg))
