"""Smoke tests. Run against the local-echo provider so CI doesn't need API keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eval_harness.config import EvalConfig, ModelSpec
from ai_eval_harness.cost import CostTracker, RequestRecord
from ai_eval_harness.loaders import load_rows
from ai_eval_harness.runner import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "rag-baseline.yaml"
EXAMPLE_DATA = REPO_ROOT / "examples" / "data" / "rag-baseline.jsonl"


def test_example_config_loads() -> None:
    cfg = EvalConfig.from_yaml(EXAMPLE_CONFIG)
    assert cfg.name == "rag-baseline"
    assert len(cfg.models) == 3
    assert any(m.provider == "local" for m in cfg.models)


def test_example_dataset_has_rows() -> None:
    cfg = EvalConfig.from_yaml(EXAMPLE_CONFIG)
    rows = load_rows(cfg.dataset)
    assert len(rows) == 10
    assert all("question" in r and "ground_truth" in r and "contexts" in r for r in rows)


def test_unique_model_names_required() -> None:
    with pytest.raises(ValueError, match="unique"):
        EvalConfig(
            name="dup",
            dataset={"source": str(EXAMPLE_DATA)},
            models=[
                ModelSpec(
                    name="same",
                    provider="local",
                    model_id="echo",
                    input_cost_per_mtok=0,
                    output_cost_per_mtok=0,
                ),
                ModelSpec(
                    name="same",
                    provider="local",
                    model_id="echo",
                    input_cost_per_mtok=0,
                    output_cost_per_mtok=0,
                ),
            ],
        )


def test_cost_tracker_computes_usd() -> None:
    model = ModelSpec(
        name="m",
        provider="local",
        model_id="echo",
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=30.0,
    )
    tracker = CostTracker([model])
    tracker.record(
        RequestRecord("m", input_tokens=1_000_000, output_tokens=500_000, latency_ms=120.0)
    )
    [summary] = tracker.summary()
    # 1M input * $10 + 0.5M output * $30 = $10 + $15 = $25
    assert summary.cost_usd == pytest.approx(25.0)
    assert summary.requests == 1


def test_local_only_run_works_without_keys() -> None:
    """Drop the API-backed models and run the local-echo path end-to-end.
    Verifies that the harness produces a usable artifact without external creds."""
    full = EvalConfig.from_yaml(EXAMPLE_CONFIG)
    local_only = EvalConfig(
        name=full.name + "-local-smoke",
        description=full.description,
        dataset=full.dataset,
        retrieval=full.retrieval,
        models=[m for m in full.models if m.provider == "local"],
        metrics=full.metrics,
    )
    result = run_eval(local_only, score=False)
    assert result.rows == 10
    assert len(result.outputs) == 10
    [summary] = result.cost
    assert summary.requests == 10
    assert summary.failures == 0
    assert summary.cost_usd == 0.0


def test_result_serializes_to_json(tmp_path: Path) -> None:
    cfg = EvalConfig.from_yaml(EXAMPLE_CONFIG)
    local_only = EvalConfig(
        name=cfg.name + "-serialize",
        dataset=cfg.dataset,
        models=[m for m in cfg.models if m.provider == "local"],
    )
    result = run_eval(local_only, score=False)
    path = result.write(tmp_path)
    assert path.exists()
    payload = path.read_text()
    assert "rag-baseline-serialize" in payload
    assert '"outputs"' in payload
    assert path.with_suffix(".md").exists()


def test_qwen_comparison_config_has_baseline_and_rag() -> None:
    cfg = EvalConfig.from_yaml(REPO_ROOT / "configs" / "qwen-rag-smoke.yaml")
    rows = load_rows(cfg.dataset)
    assert len(rows) == 21
    assert all(row["contexts"] == [] for row in rows)
    assert cfg.retrieval.strategy == "bm25"
    assert [model.use_retrieval for model in cfg.models] == [False, True]
