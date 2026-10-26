"""Dataset loaders. Local CSV/JSONL and HuggingFace datasets supported."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from ai_eval_harness.config import DatasetSpec


def load_rows(spec: DatasetSpec) -> list[dict[str, object]]:
    """Load and normalize dataset rows.

    Returns a list of dicts with keys: question, ground_truth, contexts.
    Raises FileNotFoundError if a local file source is missing.
    """
    raw_rows = list(_iter_raw_rows(spec))

    out: list[dict[str, object]] = []
    cols = spec.columns
    for raw in raw_rows:
        row = {
            "id": raw.get("id", f"row-{len(out) + 1}"),
            "language": raw.get("language", ""),
            "question": raw.get(cols["question"]),
            "ground_truth": raw.get(cols["ground_truth"]),
            "accepted_answers": _coerce_contexts(raw.get("accepted_answers")),
            "source_ids": _coerce_contexts(raw.get("source_ids")),
            "contexts": (
                _coerce_contexts(raw.get(cols["contexts"])) if spec.include_contexts else []
            ),
        }
        if row["question"] is None or row["ground_truth"] is None:
            continue  # malformed row
        out.append(row)
        if spec.max_rows and len(out) >= spec.max_rows:
            break
    return out


def _iter_raw_rows(spec: DatasetSpec) -> Iterator[dict[str, object]]:
    source = spec.source
    p = Path(source)

    if p.exists():
        if p.suffix == ".jsonl":
            yield from _read_jsonl(p)
        elif p.suffix == ".csv":
            yield from _read_csv(p)
        else:
            raise ValueError(f"unsupported local file type: {p.suffix} ({p})")
        return

    # Not a local file → HuggingFace dataset id (org/name)
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Loading HuggingFace datasets requires the 'ragas' extra: "
            "pip install 'ai-eval-harness[ragas]'"
        ) from e

    ds = load_dataset(source, split=spec.split)
    for row in ds:
        yield dict(row)


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> Iterator[dict[str, object]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield dict(row)


def _coerce_contexts(value: object) -> list[str]:
    """Contexts must be a list of strings for Ragas. Coerce common shapes."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        # JSON-encoded list or single-string context
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return [s]
    return [str(value)]
