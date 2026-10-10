"""Stable Diffusion 3 / 3.5 foundation model manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class SD3Adapter(BaseManifoldAdapter):
    """Adapter optimized for SD3.5 multi-text-encoder (T5-XXL + dual CLIP) distribution."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.SD3

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        parts = [slots.subject]

        if slots.subject_attributes:
            parts.append(", ".join(slots.subject_attributes[:3]))

        if slots.environment:
            parts.append(f"set in {slots.environment}")

        if slots.lighting:
            parts.append(slots.lighting)

        if slots.camera_framing:
            parts.append(slots.camera_framing)

        if slots.color_palette:
            parts.append(slots.color_palette)

        prompt = ", ".join(parts) + ", sharp focus, photorealistic rendering"
        metadata = {
            "target": self.target.value,
            "encoder": "t5_xxl_clip_g_clip_l",
            "syntax_mode": "hybrid_descriptive_clauses",
        }
        return prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        base_neg = [
            "blurry", "distorted anatomy", "bad hands", "low resolution",
            "oversaturated", "amateur", "glitch", "jpeg artifacts"
        ]
        if slots.explicit_negatives:
            base_neg = slots.explicit_negatives + base_neg
        return ", ".join(dict.fromkeys(base_neg))
