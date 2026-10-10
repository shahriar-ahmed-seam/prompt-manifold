"""SDXL foundation model manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget, MediumType


class SDXLAdapter(BaseManifoldAdapter):
    """Adapter optimized for SDXL dual-text-encoder token matrix."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.SDXL

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        tokens = []

        # SDXL benefits from weighted primary subject tokens
        tokens.append(f"({slots.subject}:1.2)")

        if slots.subject_attributes:
            tokens.extend(slots.subject_attributes[:3])

        if slots.environment:
            tokens.append(slots.environment)

        if slots.lighting:
            tokens.append(slots.lighting)

        if slots.camera_framing:
            tokens.append(slots.camera_framing)

        if slots.style_keywords:
            tokens.extend(slots.style_keywords)

        # Append quality tokens calibrated for SDXL
        if slots.medium == MediumType.PHOTOGRAPHY:
            tokens.extend(["raw photo", "8k uhd", "dslr quality", "sharp focus"])
        else:
            tokens.extend(["highly detailed", "sharp focus", "masterful artwork"])

        prompt = ", ".join(tokens)
        metadata = {
            "target": self.target.value,
            "encoder": "clip_l_openclip_g",
            "syntax_mode": "weighted_comma_tokens",
            "token_estimate": len(prompt.split()),
        }
        return prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        base_neg = [
            "deformed", "bad anatomy", "bad proportions", "blurry",
            "cloned face", "disfigured", "extra limbs", "mutated",
            "poorly drawn face", "low quality", "jpeg artifacts",
            "signature", "watermark", "oversaturated"
        ]
        if slots.explicit_negatives:
            base_neg = slots.explicit_negatives + base_neg
        return ", ".join(dict.fromkeys(base_neg))
