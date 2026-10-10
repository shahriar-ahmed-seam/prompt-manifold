"""Core PromptManifold projection engine."""

import time
from typing import List, Optional, Tuple

from prompt_manifold.adapters import AdapterRegistry, get_default_registry
from prompt_manifold.core.intent import IntentExtractor
from prompt_manifold.schemas import (
    EnhancedPromptResult,
    ExpansionConfig,
    IntentSlots,
    MediumType,
)


class PromptManifoldEngine:
    """Inference-time prompt distribution alignment engine.

    Transforms sparse user input into high-dimensional in-distribution representations
    matching the training manifold of the target foundation model.
    """

    def __init__(
        self,
        extractor: Optional[IntentExtractor] = None,
        registry: Optional[AdapterRegistry] = None,
    ):
        self.extractor = extractor or IntentExtractor()
        self.registry = registry or get_default_registry()

    def enhance(
        self,
        raw_prompt: str,
        config: Optional[ExpansionConfig] = None,
    ) -> EnhancedPromptResult:
        """Project a sparse user prompt onto the target training manifold."""
        start_time = time.perf_counter()
        cfg = config or ExpansionConfig()

        # Step 1: Syntactic intent slot extraction
        slots = self.extractor.extract(raw_prompt)

        # Apply manual slot overrides if provided
        if cfg.custom_seed_slots:
            for k, v in cfg.custom_seed_slots.items():
                if hasattr(slots, k) and v is not None:
                    setattr(slots, k, v)

        # Step 2: Manifold slot completion
        completed_slots = self._complete_manifold_slots(slots, cfg)

        # Step 3: Format in-distribution prompt according to target model manifold
        enhanced_prompt, adapter_meta = self._format_target_prompt(completed_slots, cfg)

        # Step 4: Synthesize dual negative prompt constraints if configured
        negative_prompt = (
            self._synthesize_negative_prompt(completed_slots, cfg)
            if cfg.generate_negative
            else None
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Compute distribution and telemetry metrics
        in_dist_score = self._compute_in_distribution_score(enhanced_prompt, completed_slots)
        raw_word_count = max(1, len(raw_prompt.split()))
        enhanced_word_count = max(1, len(enhanced_prompt.split()))
        expansion_ratio = round(enhanced_word_count / raw_word_count, 2)
        drift_score = self._compute_drift_score(raw_prompt, completed_slots.subject)
        slot_coverage = self._compute_slot_coverage(completed_slots)

        return EnhancedPromptResult(
            raw_input=raw_prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            target=cfg.target_model,
            slots=completed_slots,
            in_distribution_score=in_dist_score,
            expansion_ratio=max(1.0, expansion_ratio),
            semantic_drift_score=drift_score,
            slot_coverage_score=slot_coverage,
            latency_ms=round(latency_ms, 2),
            adapter_metadata=adapter_meta,
        )

    def enhance_batch(
        self,
        raw_prompts: List[str],
        config: Optional[ExpansionConfig] = None,
    ) -> List[EnhancedPromptResult]:
        """Process batch of sparse prompts through the projection manifold."""
        return [self.enhance(p, config) for p in raw_prompts]

    def _complete_manifold_slots(self, slots: IntentSlots, cfg: ExpansionConfig) -> IntentSlots:
        environment = slots.environment
        if not environment:
            environment = self._infer_environment(slots.subject, slots.medium)

        lighting = slots.lighting
        if not lighting:
            lighting = self._infer_lighting(slots.medium, environment)

        camera = slots.camera_framing
        if not camera and cfg.inject_optical_parameters:
            camera = self._infer_camera(slots.medium, slots.subject)

        palette = (
            slots.color_palette
            or "naturalistic balanced color palette with authentic tonal gradations"
        )
        atmosphere = slots.atmosphere or "tactile atmosphere with spatial separation and depth"

        return IntentSlots(
            subject=slots.subject,
            subject_attributes=slots.subject_attributes,
            environment=environment,
            lighting=lighting,
            medium=slots.medium,
            camera_framing=camera,
            color_palette=palette,
            atmosphere=atmosphere,
            explicit_negatives=slots.explicit_negatives,
            style_keywords=slots.style_keywords,
            preserved_tokens=slots.preserved_tokens,
        )

    def _infer_environment(self, subject: str, medium: MediumType) -> str:
        sub = subject.lower()
        if "bear" in sub or "book" in sub or "reading" in sub:
            return "rustic cozy timber interior with mahogany bookshelf and soft woven blanket"
        elif any(k in sub for k in ("portrait", "person", "fashion")):
            return "minimalist architectural interior with textured stone wall and negative space"
        elif any(k in sub for k in ("cyber", "city", "neon", "street")):
            return (
                "rain-slicked metropolitan street flanked by towering futuristic facades "
                "and wet reflective asphalt"
            )
        elif any(k in sub for k in ("shoe", "product", "bottle", "watch")):
            return "monolithic matte slate podium set against an elegant gradient studio backdrop"
        elif any(k in sub for k in ("landscape", "mountain", "forest")):
            return "sweeping alpine wilderness bordered by glacial river and pine canopy"
        return "cohesive atmospheric environment with naturalistic spatial depth"

    def _infer_lighting(self, medium: MediumType, environment: str) -> str:
        env = environment.lower()
        if any(k in env for k in ("rain", "cyber", "night")):
            return (
                "volumetric neon ambient light with puddle reflections and cold cyan rim highlights"
            )
        elif any(k in env for k in ("cozy", "timber", "interior")):
            return "soft directional morning sunlight streaming through nearby window"
        elif any(k in env for k in ("podium", "studio")):
            return "softbox key light with subtle fill and crisp edge-defining rim lighting"
        return "diffused natural daylight with soft organic shadows"

    def _infer_camera(self, medium: MediumType, subject: str) -> str:
        if medium in (MediumType.PHOTOGRAPHY, MediumType.CINEMATIC):
            sub = subject.lower()
            if any(k in sub for k in ("product", "shoe", "watch", "bear")):
                return (
                    "medium close-up, 85mm f/2.0 prime lens, "
                    "shallow depth of field, sharp foreground"
                )
            elif any(k in sub for k in ("street", "landscape")):
                return (
                    "cinematic 35mm wide anamorphic lens, "
                    "subtle distortion, expansive field of view"
                )
            return "sharp 50mm f/1.8 lens, natural perspective, balanced eye-level framing"
        return "centered balanced framing with crisp focal hierarchy"

    def _format_target_prompt(
        self, slots: IntentSlots, cfg: ExpansionConfig
    ) -> Tuple[str, dict]:
        """Synthesize final prompt adapted to target foundation model syntax."""
        try:
            adapter = self.registry.get(cfg.target_model)
            return adapter.format_prompt(slots, cfg)
        except KeyError:
            # Fallback formatting if target is not registered
            clauses = [slots.subject]
            if slots.subject_attributes:
                clauses.append(", ".join(slots.subject_attributes[:3]))
            if slots.environment:
                clauses.append(f"in {slots.environment}")
            if slots.lighting:
                clauses.append(slots.lighting)
            return ". ".join(clauses) + ".", {"syntax_mode": "fallback_default"}

    def _synthesize_negative_prompt(self, slots: IntentSlots, cfg: ExpansionConfig) -> str:
        """Synthesize dual negative prompt constraints."""
        try:
            adapter = self.registry.get(cfg.target_model)
            return adapter.synthesize_negative(slots, cfg)
        except KeyError:
            base_neg = ["blurry", "low quality", "distorted", "watermark"]
            if slots.explicit_negatives:
                base_neg = slots.explicit_negatives + base_neg
            return ", ".join(dict.fromkeys(base_neg))

    def _compute_in_distribution_score(self, prompt: str, slots: IntentSlots) -> float:
        """Estimate in-distribution proximity index based on syntactic token density."""
        words = prompt.split()
        length = len(words)
        if length < 10:
            len_score = 0.2
        elif length < 30:
            len_score = 0.5 + (length - 10) * 0.015
        elif length <= 120:
            len_score = 0.8 + min(0.2, (length - 30) * 0.0022)
        else:
            len_score = 0.95

        slot_score = self._compute_slot_coverage(slots)
        score = 0.5 * len_score + 0.5 * slot_score
        return round(min(1.0, max(0.1, score)), 3)

    def _compute_drift_score(self, raw_prompt: str, extracted_subject: str) -> float:
        """Measure semantic drift penalty. 0.0 indicates complete alignment with user intent."""
        raw_words = set(w.lower().strip(".,;:!'\"") for w in raw_prompt.split() if len(w) > 2)
        if not raw_words:
            return 0.0
        sub_words = set(
            w.lower().strip(".,;:!'\"") for w in extracted_subject.split() if len(w) > 2
        )
        overlap = raw_words.intersection(sub_words)
        if not overlap:
            return 0.15
        ratio = len(overlap) / len(raw_words)
        drift = max(0.0, 1.0 - ratio) * 0.2
        return round(drift, 3)

    def _compute_slot_coverage(self, slots: IntentSlots) -> float:
        """Compute the completeness of syntactic dimension slots."""
        total_slots = 6
        present = 1
        if slots.environment:
            present += 1
        if slots.lighting:
            present += 1
        if slots.camera_framing:
            present += 1
        if slots.color_palette:
            present += 1
        if slots.atmosphere:
            present += 1
        return round(present / total_slots, 3)
