"""DALL-E 3 foundation model manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class Dalle3Adapter(BaseManifoldAdapter):
    """Adapter optimized for DALL-E 3 narrative world-building distribution."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.DALLE3

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        narrative = [f"A detailed depiction of {slots.subject}."]

        if slots.subject_attributes:
            attrs = ", ".join(slots.subject_attributes)
            narrative.append(f"The subject prominently features {attrs}.")

        if slots.environment:
            narrative.append(f"The surrounding background reveals a {slots.environment}.")

        if slots.lighting:
            narrative.append(f"Lighting is defined by {slots.lighting}.")

        if slots.camera_framing:
            narrative.append(f"Viewed from a {slots.camera_framing}.")

        prompt = " ".join(narrative)
        metadata = {
            "target": self.target.value,
            "engine": "dalle_3_narrative",
        }
        return prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        # DALL-E 3 handles negative concepts conversationally
        neg_items = ["blurry", "low quality", "unrealistic anatomy", "watermark"]
        if slots.explicit_negatives:
            neg_items = slots.explicit_negatives + neg_items
        return ", ".join(dict.fromkeys(neg_items))
