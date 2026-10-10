"""Formal schemas and data protocols for the PromptManifold engine."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ManifoldTarget(str, Enum):
    """Target generative model distribution manifolds."""
    FLUX = "flux"
    SDXL = "sdxl"
    SD3 = "sd3"
    MIDJOURNEY = "midjourney"
    DALLE3 = "dalle3"
    UNIVERSAL_VISION = "universal_vision"
    LLM_COGNITIVE = "llm_cognitive"


class MediumType(str, Enum):
    """Syntactic medium classifications."""
    PHOTOGRAPHY = "photography"
    DIGITAL_ART = "digital_art"
    CINEMATIC = "cinematic"
    CONCEPT_ART = "concept_art"
    THREE_D_RENDER = "3d_render"
    ANIME_MANGA = "anime_manga"
    OIL_PAINTING = "oil_painting"
    VECTOR_FLAT = "vector_flat"


class IntentSlots(BaseModel):
    """Extracted semantic slots from sparse user prompts."""
    subject: str = Field(..., description="Foundational semantic anchor or entity.")
    subject_attributes: List[str] = Field(
        default_factory=list, description="Descriptive adjectives and morphology."
    )
    environment: Optional[str] = Field(
        None, description="Spatial background and atmospheric environment."
    )
    lighting: Optional[str] = Field(
        None, description="Illumination schema and light source parameters."
    )
    medium: MediumType = Field(
        default=MediumType.PHOTOGRAPHY, description="Primary visual artistic medium."
    )
    camera_framing: Optional[str] = Field(
        None, description="Focal length, camera angle, and optical parameters."
    )
    color_palette: Optional[str] = Field(
        None, description="Color harmony and tonal distribution."
    )
    atmosphere: Optional[str] = Field(
        None, description="Emotional or environmental mood."
    )
    explicit_negatives: List[str] = Field(
        default_factory=list, description="Explicit exclusions parsed from input."
    )
    style_keywords: List[str] = Field(
        default_factory=list, description="Artistic styles or movement references."
    )
    preserved_tokens: List[str] = Field(
        default_factory=list, description="Literal trigger words that must not be mutated."
    )


class ExpansionConfig(BaseModel):
    """Configuration governing inference-time prompt manifold projection."""
    target_model: ManifoldTarget = Field(
        default=ManifoldTarget.FLUX, description="Target foundation model."
    )
    detail_level: int = Field(
        default=3, ge=1, le=5, description="Expansion density scale 1 (sparse) to 5 (dense)."
    )
    preserve_exact_subject: bool = Field(
        default=True, description="Enforce strict semantic retention of core subject."
    )
    generate_negative: bool = Field(
        default=True, description="Synthesize paired negative constraint vector."
    )
    max_tokens: int = Field(
        default=160, ge=20, le=512, description="Maximum token allocation budget."
    )
    inject_optical_parameters: bool = Field(
        default=True, description="Include photorealistic lens/lighting slots."
    )
    custom_seed_slots: Optional[Dict[str, Any]] = Field(
        None, description="Manual slot overrides if provided."
    )


class EnhancedPromptResult(BaseModel):
    """Complete output payload of the PromptManifold projection layer."""
    raw_input: str = Field(..., description="Original sparse user query.")
    enhanced_prompt: str = Field(..., description="In-distribution projected prompt.")
    negative_prompt: Optional[str] = Field(
        None, description="Paired negative distribution constraint."
    )
    target: ManifoldTarget = Field(..., description="Target foundation model distribution.")
    slots: IntentSlots = Field(..., description="Semantic slots used during projection.")
    in_distribution_score: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated in-distribution proximity index."
    )
    expansion_ratio: float = Field(
        ..., ge=1.0, description="Ratio of enhanced prompt length to raw input length."
    )
    semantic_drift_score: float = Field(
        ..., ge=0.0, le=1.0, description="Intent deviation metric (0=zero drift, 1=corrupted)."
    )
    slot_coverage_score: float = Field(
        ..., ge=0.0, le=1.0, description="Completeness of target syntactic slots."
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="End-to-end projection latency in milliseconds."
    )
    adapter_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Target-specific execution metadata."
    )
