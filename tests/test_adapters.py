"""Unit tests for foundation model manifold adapters."""

import pytest

from prompt_manifold.adapters import (
    AdapterRegistry,
    Dalle3Adapter,
    FluxAdapter,
    MidjourneyAdapter,
    SD3Adapter,
    SDXLAdapter,
    UniversalVisionAdapter,
    get_default_registry,
)
from prompt_manifold.core.intent import IntentExtractor
from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget


@pytest.fixture
def extractor():
    return IntentExtractor()


@pytest.fixture
def teddy_slots(extractor):
    return extractor.extract("a teddy bear reading a book")


def test_flux_adapter_natural_prose(teddy_slots):
    """Flux adapter should generate natural narrative sentences without quality tag salad."""
    adapter = FluxAdapter()
    cfg = ExpansionConfig(target_model=ManifoldTarget.FLUX)
    prompt, meta = adapter.format_prompt(teddy_slots, cfg)

    assert "bear" in prompt.lower()
    assert "book" in prompt.lower()
    assert meta["target"] == "flux"
    assert meta["avoid_quality_tokens"] is True
    # Ensure no naive tag spam
    assert "trending on artstation" not in prompt.lower()
    assert "masterpiece" not in prompt.lower()

    neg = adapter.synthesize_negative(teddy_slots, ExpansionConfig())
    assert "blurry" in neg


def test_sdxl_adapter_syntax_weighting(teddy_slots):
    """SDXL adapter should include syntax weighting parentheses and dual-CLIP tokens."""
    adapter = SDXLAdapter()
    cfg = ExpansionConfig(target_model=ManifoldTarget.SDXL)
    prompt, meta = adapter.format_prompt(teddy_slots, cfg)

    assert ":1.2)" in prompt
    assert "8k" in prompt.lower() or "photo" in prompt.lower()
    assert meta["target"] == "sdxl"

    neg = adapter.synthesize_negative(teddy_slots, ExpansionConfig())
    assert "deformed" in neg
    assert "bad anatomy" in neg


def test_midjourney_adapter_parameter_flags(teddy_slots):
    """Midjourney adapter must append CLI parameters (--v 6.1, --ar, --style raw)."""
    adapter = MidjourneyAdapter()
    cfg = ExpansionConfig(target_model=ManifoldTarget.MIDJOURNEY)
    prompt, meta = adapter.format_prompt(teddy_slots, cfg)

    assert "--v 6.1" in prompt
    assert "--style raw" in prompt
    assert "--ar 16:9" in prompt
    assert "--stylize" in prompt
    assert meta["target"] == "midjourney"

    neg = adapter.synthesize_negative(teddy_slots, ExpansionConfig())
    assert "--no" in neg


def test_sd3_adapter_t5_clip_hybrid(teddy_slots):
    """SD3 adapter should combine descriptive clauses with sharp focus anchoring."""
    adapter = SD3Adapter()
    cfg = ExpansionConfig(target_model=ManifoldTarget.SD3)
    prompt, meta = adapter.format_prompt(teddy_slots, cfg)

    assert "bear" in prompt.lower()
    assert meta["target"] == "sd3"

    neg = adapter.synthesize_negative(teddy_slots, ExpansionConfig())
    assert "blurry" in neg


def test_dalle3_adapter_narrative_style(teddy_slots):
    """DALL-E 3 adapter should produce descriptive narrative world-building."""
    adapter = Dalle3Adapter()
    cfg = ExpansionConfig(target_model=ManifoldTarget.DALLE3)
    prompt, meta = adapter.format_prompt(teddy_slots, cfg)

    assert prompt.startswith("A detailed depiction of")
    assert meta["target"] == "dalle3"


def test_adapter_registry_lookup():
    """Verify registry lookups, default mappings, and fallback behavior."""
    reg = get_default_registry()

    assert isinstance(reg.get(ManifoldTarget.FLUX), FluxAdapter)
    assert isinstance(reg.get(ManifoldTarget.SDXL), SDXLAdapter)
    assert isinstance(reg.get(ManifoldTarget.SD3), SD3Adapter)
    assert isinstance(reg.get(ManifoldTarget.MIDJOURNEY), MidjourneyAdapter)
    assert isinstance(reg.get(ManifoldTarget.DALLE3), Dalle3Adapter)
    assert isinstance(reg.get(ManifoldTarget.UNIVERSAL_VISION), UniversalVisionAdapter)

    custom_reg = AdapterRegistry()
    with pytest.raises(KeyError):
        custom_reg.get(ManifoldTarget.FLUX)
