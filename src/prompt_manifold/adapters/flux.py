"""Flux foundation model manifold adapter."""

from typing import Any, Dict, Tuple

from prompt_manifold.adapters.base import BaseManifoldAdapter
from prompt_manifold.schemas import ExpansionConfig, IntentSlots, ManifoldTarget


class FluxAdapter(BaseManifoldAdapter):
    """Adapter optimized for Flux (BFL) dense natural text distribution."""

    @property
    def target(self) -> ManifoldTarget:
        return ManifoldTarget.FLUX

    def format_prompt(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> Tuple[str, Dict[str, Any]]:
        sentences = []

        # 1. Subject description (fluid natural sentence)
        subj = slots.subject
        if slots.subject_attributes:
            attrs = ", ".join(slots.subject_attributes[:3])
            sentences.append(f"{subj.capitalize()} with {attrs}.")
        else:
            sentences.append(f"{subj.capitalize()}.")

        # 2. Environmental context
        if slots.environment:
            sentences.append(f"The scene takes place in a {slots.environment}.")

        # 3. Illumination and optical depth
        lighting_and_cam = []
        if slots.lighting:
            lighting_and_cam.append(f"illuminated by {slots.lighting}")
        if slots.camera_framing:
            lighting_and_cam.append(f"captured with {slots.camera_framing}")

        if lighting_and_cam:
            sentences.append(f"The composition is {' and '.join(lighting_and_cam)}.")

        # 4. Color harmony and atmosphere
        if slots.color_palette or slots.atmosphere:
            notes = []
            if slots.color_palette:
                notes.append(slots.color_palette)
            if slots.atmosphere:
                notes.append(slots.atmosphere)
            sentences.append(f"The visual tone showcases {', '.join(notes)}.")

        prompt = " ".join(sentences)
        metadata = {
            "target": self.target.value,
            "encoder": "t5_xxl_clip_l",
            "style_strategy": "natural_narrative_prose",
            "avoid_quality_tokens": True,
        }
        return prompt, metadata

    def synthesize_negative(
        self, slots: IntentSlots, config: ExpansionConfig
    ) -> str:
        # Flux uses guidance distilation where negative prompts are generally minimal
        base_neg = ["blurry", "low contrast", "distorted details", "watermark", "signature"]
        if slots.explicit_negatives:
            base_neg = slots.explicit_negatives + base_neg
        return ", ".join(dict.fromkeys(base_neg))
