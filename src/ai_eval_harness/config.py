"""Eval config schema.

Configs are YAML files with three top-level sections: dataset, retrieval,
and models. Anything not in this schema is rejected at load-time so configs
stay reproducible across versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelSpec(BaseModel):
    """One model under eval. Cost is per-million tokens, in USD."""

    name: str
    provider: Literal["anthropic", "openai", "ollama", "local"]
    model_id: str
    input_cost_per_mtok: float = Field(ge=0)
    output_cost_per_mtok: float = Field(ge=0)
    max_tokens: int = 1024
    temperature: float = 0.0
    use_retrieval: bool = True


class RetrievalSpec(BaseModel):
    """Retrieval pipeline under eval."""

    strategy: Literal["precomputed", "bm25"] = "precomputed"
    corpus_source: str | None = None
    top_k: int = Field(default=3, ge=1)


class DatasetSpec(BaseModel):
    """Dataset under eval. Local CSV/JSONL or a HuggingFace dataset id."""

    source: str
    """Path to a local file (.csv, .jsonl) or a HF dataset id (org/name)."""

    split: str = "test"
    """HF split. Ignored for local files."""

    max_rows: int | None = None
    """Cap rows for fast iteration. None = use full split."""

    include_contexts: bool = True
    """Set false to measure the same model without retrieval context."""

    columns: dict[str, str] = Field(
        default_factory=lambda: {
            "question": "question",
            "ground_truth": "ground_truth",
            "contexts": "contexts",
        }
    )
    """Map of schema-column → source-column. Lets you reuse any dataset shape."""


class MetricsSpec(BaseModel):
    """Which metrics to compute. Default = all Ragas v0.2 RAG-core metrics."""

    ragas: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    promptfoo: list[str] = Field(default_factory=list)
    """Optional promptfoo eval names. Wired in v0.1 but executed only if the
    promptfoo binary is on PATH (npm i -g promptfoo)."""


class EvalConfig(BaseModel):
    """Top-level eval config. Loaded from YAML, validated, then handed to the
    runner. Every published Paiteq benchmark on getwidget.dev/benchmarks/ uses
    one of these configs verbatim — that's the reproducibility contract."""

    name: str
    """Slug used in result paths. e.g. "rag-2026-q2"."""

    description: str = ""
    dataset: DatasetSpec
    retrieval: RetrievalSpec = Field(default_factory=RetrievalSpec)
    models: list[ModelSpec]
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)

    @field_validator("models")
    @classmethod
    def _at_least_one_model(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        if not v:
            raise ValueError("at least one model required under `models`")
        # Ensure unique names — results keyed by model.name
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError(f"model names must be unique; got {names}")
        return v

    @field_validator("retrieval")
    @classmethod
    def _corpus_required_for_bm25(cls, value: RetrievalSpec) -> RetrievalSpec:
        if value.strategy == "bm25" and not value.corpus_source:
            raise ValueError("retrieval.corpus_source is required for bm25")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"config {path} did not parse to a mapping")
        return cls.model_validate(raw)
