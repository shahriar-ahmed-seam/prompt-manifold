"""Unit tests for distribution metrics, drift guard, and benchmark runner."""

import pytest

from prompt_manifold.eval.benchmark import BenchmarkRunner
from prompt_manifold.metrics.distribution import DistributionEvaluator
from prompt_manifold.metrics.drift import SemanticDriftGuard
from prompt_manifold.schemas import IntentSlots, MediumType


@pytest.fixture
def drift_guard():
    return SemanticDriftGuard()


@pytest.fixture
def dist_evaluator():
    return DistributionEvaluator()


def test_semantic_drift_exact_preservation(drift_guard):
    """When enhanced prompt fully retains user subject entities, drift should be minimal."""
    raw = "a teddy bear reading a book"
    enh = (
        "A cute plush teddy bear with vintage stitching sits quietly reading an antique book "
        "beside a glowing fireplace in a rustic library."
    )
    drift = drift_guard.compute_drift(raw, enh)
    assert drift <= 0.1
    assert drift_guard.verify_entity_retention(raw, enh) is True


def test_semantic_drift_hijacking_penalty(drift_guard):
    """When user prompt subject is entirely omitted, drift penalty should be high."""
    raw = "a red sports car"
    hijacked = "A tranquil green meadow filled with yellow sunflowers on a summer morning."
    drift = drift_guard.compute_drift(raw, hijacked)
    assert drift > 0.5
    assert drift_guard.verify_entity_retention(raw, hijacked) is False


def test_distribution_evaluator_metrics(dist_evaluator):
    """Verify in-distribution score, slot coverage, and quality index calculations."""
    slots = IntentSlots(
        subject="teddy bear",
        environment="cozy library",
        lighting="soft warm window light",
        medium=MediumType.PHOTOGRAPHY,
        camera_framing="close-up 85mm lens",
        color_palette="earth tones",
        atmosphere="intimate quiet",
    )
    raw = "teddy bear"
    enh = "A plush teddy bear in a cozy library with soft warm window light."

    metrics = dist_evaluator.compute_all_metrics(
        raw_prompt=raw,
        enhanced_prompt=enh,
        slots=slots,
        drift_score=0.0,
    )

    assert 0.0 <= metrics["in_distribution_score"] <= 1.0
    assert metrics["expansion_ratio"] >= 1.0
    assert metrics["slot_coverage_score"] == 1.0
    assert 0.0 <= metrics["manifold_quality_index"] <= 1.0


def test_benchmark_runner_suite():
    """Verify that BenchmarkRunner executes all scenarios and generates comparative telemetry."""
    runner = BenchmarkRunner()
    report = runner.run_suite()

    assert report["scenario_count"] >= 5
    assert report["mean_enhanced_id_score"] > report["mean_naive_id_score"]
    assert report["win_rate_percentage"] == 100.0
    assert report["mean_latency_ms"] < 50.0  # sub-50ms heuristic projection
