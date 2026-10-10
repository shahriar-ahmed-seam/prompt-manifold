"""Unit tests for the PromptManifold core engine and intent extractor."""

import pytest

from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.core.intent import IntentExtractor
from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget


@pytest.fixture
def engine():
    return PromptManifoldEngine()


def test_sparse_teddy_bear_infographic_case(engine):
    """Verify that 'a teddy bear reading a book' expands onto the prompt manifold correctly."""
    raw = "a teddy bear reading a book"
    res = engine.enhance(raw, ExpansionConfig(target_model=ManifoldTarget.FLUX))

    assert res.raw_input == raw
    assert "bear" in res.enhanced_prompt.lower()
    assert "book" in res.enhanced_prompt.lower()
    assert res.slots.subject != ""
    assert res.in_distribution_score > 0.7
    assert res.expansion_ratio > 2.0
    assert res.semantic_drift_score <= 0.1
    assert res.slot_coverage_score >= 0.8
    assert res.negative_prompt is not None
    assert "blurry" in res.negative_prompt


def test_single_word_sparse_input(engine):
    """Verify that an ultra-sparse single word like 'meat' or 'chair' is safely mapped."""
    raw = "meat"
    res = engine.enhance(raw, ExpansionConfig(target_model=ManifoldTarget.SDXL))

    assert "meat" in res.enhanced_prompt.lower()
    assert res.expansion_ratio > 3.0
    assert res.in_distribution_score >= 0.6
    assert res.latency_ms >= 0.0


def test_empty_and_whitespace_edge_cases(engine):
    """Verify that empty inputs gracefully fallback without crashing."""
    res_empty = engine.enhance("")
    assert res_empty.enhanced_prompt != ""
    assert res_empty.in_distribution_score > 0.5

    res_spaces = engine.enhance("   \n\t   ")
    assert res_spaces.enhanced_prompt != ""


def test_explicit_negation_extraction(engine):
    """Verify user explicit negations (e.g. 'no blur') are parsed into negative prompt."""
    raw = "cyberpunk street at night, no blur, without text"
    res = engine.enhance(raw)

    assert "street" in res.enhanced_prompt.lower()
    assert res.negative_prompt is not None
    assert "blur" in res.negative_prompt.lower() or "text" in res.negative_prompt.lower()


def test_preserved_trigger_tokens(engine):
    """Verify that quoted trigger tokens are preserved."""
    raw = 'a portrait of "cyber_warrior_v2" in rain'
    extractor = IntentExtractor()
    slots = extractor.extract(raw)

    assert "cyber_warrior_v2" in slots.preserved_tokens


def test_batch_enhancement(engine):
    """Verify that batch enhancement processes multiple queries deterministically."""
    prompts = [
        "red sports car",
        "futuristic skyscraper",
        "macro flower with water droplets",
    ]
    results = engine.enhance_batch(prompts)

    assert len(results) == 3
    for r in results:
        assert len(r.enhanced_prompt) > len(r.raw_input)
        assert r.in_distribution_score > 0.6
