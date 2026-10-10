"""Universal fallback manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class UniversalVisionAdapter(BaseManifoldAdapter):
    """Universal balanced adapter for standard multimodal vision and generation systems."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.UNIVERSAL_VISION

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        parts = [slots.subject]

        if slots.subject_attributes:
            parts.append(", ".join(slots.subject_attributes[:2]))

        if slots.environment:
            parts.append(f"in {slots.environment}")

        if slots.lighting:
            parts.append(slots.lighting)

        if slots.camera_framing:
            parts.append(slots.camera_framing)

        prompt = ", ".join(parts) + "."
        metadata = {
            "target": self.target.value,
            "engine": "universal_vision",
        }
        return prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        base_neg = ["blurry", "low quality", "distorted", "watermark"]
        if slots.explicit_negatives:
            base_neg = slots.explicit_negatives + base_neg
        return ", ".join(dict.fromkeys(base_neg))
