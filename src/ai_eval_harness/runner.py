"""Eval runner. Orchestrates: load dataset → generate per model → score via
Ragas (and optionally promptfoo) → record cost → emit report."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eval_harness import promptfoo_adapter
from ai_eval_harness.config import EvalConfig, ModelSpec
from ai_eval_harness.cost import CostTracker, ModelCostSummary
from ai_eval_harness.loaders import load_rows
from ai_eval_harness.providers import get_generator
from ai_eval_harness.retrieval import BM25Retriever, load_corpus

# Default RAG-style prompt used when the dataset gives us (question, contexts)
# but no model-specific prompt template. Kept boring on purpose — eval
# harness is for comparing models, not for prompt engineering.
_DEFAULT_PROMPT_TEMPLATE = """Answer the question using only the provided sources.

Sources:
{contexts}

Question: {question}

Answer in the same language as the question. Be concise and cite supporting source IDs in square brackets. If the sources do not contain the answer, say "Not in sources."
"""

_BASELINE_PROMPT_TEMPLATE = """Answer the following question concisely using your existing knowledge.

Question: {question}
"""


@dataclass
class PerRowOutput:
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str
    model_name: str
    error: str | None = None
    row_id: str = ""
    language: str = ""
    accepted_answers: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    config: EvalConfig
    rows: int
    outputs: list[PerRowOutput]
    cost: list[ModelCostSummary]
    ragas_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """ragas_scores[model_name][metric_name] = score"""

    promptfoo_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """promptfoo_scores[model_name][metric_name] = pass_rate (0..1).
    Empty when the promptfoo binary isn't installed or metrics.promptfoo is empty."""

    promptfoo_note: str | None = None
    """Human-readable status (e.g. "promptfoo binary not on PATH; skipped").
    Set when promptfoo was requested but didn't run."""

    deterministic_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    language_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(),
            "rows": self.rows,
            "cost": [c.to_row() for c in self.cost],
            "ragas_scores": self.ragas_scores,
            "promptfoo_scores": self.promptfoo_scores,
            "promptfoo_note": self.promptfoo_note,
            "deterministic_scores": self.deterministic_scores,
            "language_scores": self.language_scores,
            "outputs": [
                {
                    "row_id": o.row_id,
                    "language": o.language,
                    "model": o.model_name,
                    "question": o.question,
                    "ground_truth": o.ground_truth,
                    "accepted_answers": o.accepted_answers,
                    "source_ids": o.source_ids,
                    "retrieved_doc_ids": o.retrieved_doc_ids,
                    "answer": o.answer,
                    "error": o.error,
                }
                for o in self.outputs
            ],
        }

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.config.name}.json"
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False))
        path.with_suffix(".md").write_text(self.to_markdown())
        return path

    def to_markdown(self) -> str:
        lines = [
            f"# {self.config.name}",
            "",
            self.config.description,
            "",
            f"Questions: {self.rows}",
            "",
            "## Aggregate results",
            "",
            "| Model | Answer-key hit | Retrieval recall@k | Correct citation | Failures |",
            "|---|---:|---:|---:|---:|",
        ]
        for model in self.config.models:
            scores = self.deterministic_scores.get(model.name, {})
            lines.append(
                "| {name} | {answer} | {retrieval} | {citation} | {failures} |".format(
                    name=model.name,
                    answer=_percent(scores.get("answer_key_hit_rate")),
                    retrieval=_percent(scores.get("retrieval_recall_at_k")),
                    citation=_percent(scores.get("correct_citation_rate")),
                    failures=_percent(scores.get("failure_rate")),
                )
            )
        lines.extend(
            [
                "",
                "## Answer-key hit rate by language",
                "",
                "| Model / language | Answer-key hit | Failures |",
                "|---|---:|---:|",
            ]
        )
        for group_name, scores in self.language_scores.items():
            lines.append(
                f"| {group_name} | {_percent(scores.get('answer_key_hit_rate'))} | "
                f"{_percent(scores.get('failure_rate'))} |"
            )
        lines.extend(["", "## Per-question outputs", ""])
        for output in self.outputs:
            lines.extend(
                [
                    f"### {output.row_id or 'row'} — {output.model_name}",
                    "",
                    f"**Question:** {output.question}",
                    "",
                    f"**Answer:** {output.answer or '(no answer)'}",
                    "",
                    f"**Expected:** {output.ground_truth}",
                    "",
                    f"**Retrieved:** {', '.join(output.retrieved_doc_ids) or 'none'}",
                    "",
                ]
            )
        return "\n".join(lines)


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def run_eval(
    config: EvalConfig,
    *,
    score: bool = True,
    work_dir: str | Path | None = None,
) -> EvalResult:
    """Run an eval end-to-end. When `score=False`, the Ragas + promptfoo
    scoring steps are skipped — useful for smoke tests that don't have
    judge-model credentials.

    `work_dir` is where promptfoo's generated config + output JSON land.
    Defaults to a per-run temp directory; pass an explicit path when you
    want to inspect the promptfoo artifacts after the run."""
    import tempfile

    rows = load_rows(config.dataset)
    retriever = _build_retriever(config)
    tracker = CostTracker(config.models)
    outputs: list[PerRowOutput] = []

    for model in config.models:
        outputs.extend(_run_one_model(model, rows, tracker, retriever, config.retrieval.top_k))

    deterministic_scores = _score_deterministic(outputs)
    language_scores = _score_by_language(outputs)

    ragas_scores: dict[str, dict[str, float]] = {}
    if score and config.metrics.ragas:
        ragas_scores = _score_with_ragas(config, outputs)

    promptfoo_scores: dict[str, dict[str, float]] = {}
    promptfoo_note: str | None = None
    if score and config.metrics.promptfoo:
        wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ai-eval-pf-"))
        pf = promptfoo_adapter.run(config, outputs, work_dir=wd / "promptfoo")
        if not pf.available:
            promptfoo_note = "promptfoo binary not on PATH; install via `npm i -g promptfoo`"
        elif pf.error:
            promptfoo_note = f"promptfoo error: {pf.error}"
        elif pf.scores:
            promptfoo_scores = pf.scores

    return EvalResult(
        config=config,
        rows=len(rows),
        outputs=outputs,
        cost=tracker.summary(),
        ragas_scores=ragas_scores,
        promptfoo_scores=promptfoo_scores,
        promptfoo_note=promptfoo_note,
        deterministic_scores=deterministic_scores,
        language_scores=language_scores,
    )


def _run_one_model(
    model: ModelSpec,
    rows: list[dict[str, object]],
    tracker: CostTracker,
    retriever: BM25Retriever | None,
    top_k: int,
) -> list[PerRowOutput]:
    gen = get_generator(model.provider)
    outs: list[PerRowOutput] = []
    for row in rows:
        row_id = str(row["id"])
        language = str(row["language"])
        question = str(row["question"])
        accepted_answers = [str(c) for c in row.get("accepted_answers", [])]  # type: ignore[arg-type]
        source_ids = [str(c) for c in row.get("source_ids", [])]  # type: ignore[arg-type]
        retrieved_docs = (
            retriever.retrieve(question, top_k, language=language)
            if model.use_retrieval and retriever
            else []
        )
        contexts = (
            [doc.prompt_text() for doc in retrieved_docs]
            if retrieved_docs
            else [str(c) for c in row.get("contexts", [])]  # type: ignore[arg-type]
            if model.use_retrieval
            else []
        )
        retrieved_doc_ids = [doc.id for doc in retrieved_docs]
        ground_truth = str(row["ground_truth"])
        prompt = (
            _DEFAULT_PROMPT_TEMPLATE.format(
                contexts="\n\n".join(contexts),
                question=question,
            )
            if contexts
            else _BASELINE_PROMPT_TEMPLATE.format(question=question)
        )
        with tracker.time(model.name) as t:
            try:
                answer, in_tok, out_tok = gen.generate(model, prompt)
                t.commit(in_tok, out_tok)
                outs.append(
                    PerRowOutput(
                        row_id=row_id,
                        language=language,
                        question=question,
                        ground_truth=ground_truth,
                        accepted_answers=accepted_answers,
                        source_ids=source_ids,
                        contexts=contexts,
                        retrieved_doc_ids=retrieved_doc_ids,
                        answer=answer,
                        model_name=model.name,
                    )
                )
            except Exception as e:  # noqa: BLE001 — log + continue
                t.commit(0, 0, error=repr(e))
                outs.append(
                    PerRowOutput(
                        row_id=row_id,
                        language=language,
                        question=question,
                        ground_truth=ground_truth,
                        accepted_answers=accepted_answers,
                        source_ids=source_ids,
                        contexts=contexts,
                        retrieved_doc_ids=retrieved_doc_ids,
                        answer="",
                        model_name=model.name,
                        error=repr(e),
                    )
                )
    return outs


def _build_retriever(config: EvalConfig) -> BM25Retriever | None:
    if config.retrieval.strategy != "bm25":
        return None
    return BM25Retriever(load_corpus(config.retrieval.corpus_source or ""))


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, re.UNICODE))


def _score_deterministic(outputs: list[PerRowOutput]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PerRowOutput]] = {}
    for output in outputs:
        grouped.setdefault(output.model_name, []).append(output)

    scores: dict[str, dict[str, float]] = {}
    for model_name, group in grouped.items():
        successful = [output for output in group if not output.error]
        if not successful:
            continue
        answer_hits = 0
        retrieval_hits = 0
        citation_hits = 0
        retrieval_rows = 0
        for output in successful:
            normalized_answer = _normalize(output.answer)
            aliases = output.accepted_answers or [output.ground_truth]
            if any(_normalize(alias) in normalized_answer for alias in aliases):
                answer_hits += 1
            if output.retrieved_doc_ids:
                retrieval_rows += 1
                gold_retrieved = set(output.source_ids) & set(output.retrieved_doc_ids)
                if gold_retrieved:
                    retrieval_hits += 1
                if _cited_doc_ids(output.answer) & gold_retrieved:
                    citation_hits += 1
        result = {
            "answer_key_hit_rate": answer_hits / len(successful),
            "failure_rate": 1 - len(successful) / len(group),
        }
        if retrieval_rows:
            result["retrieval_recall_at_k"] = retrieval_hits / retrieval_rows
            result["correct_citation_rate"] = citation_hits / retrieval_rows
        scores[model_name] = result
    return scores


def _cited_doc_ids(answer: str) -> set[str]:
    cited: set[str] = set()
    for value in re.findall(r"\[([^\]]+)\]", answer):
        normalized = re.sub(r"^source\s*:\s*", "", value.strip(), flags=re.IGNORECASE)
        cited.add(normalized)
    return cited


def _score_by_language(outputs: list[PerRowOutput]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PerRowOutput]] = {}
    for output in outputs:
        grouped.setdefault(f"{output.model_name}/{output.language or 'unknown'}", []).append(output)

    scores: dict[str, dict[str, float]] = {}
    for name, group in grouped.items():
        successful = [output for output in group if not output.error]
        answer_hits = 0
        for output in successful:
            aliases = output.accepted_answers or [output.ground_truth]
            if any(_normalize(alias) in _normalize(output.answer) for alias in aliases):
                answer_hits += 1
        scores[name] = {
            "answer_key_hit_rate": answer_hits / len(successful) if successful else 0.0,
            "failure_rate": 1 - len(successful) / len(group),
        }
    return scores


def _score_with_ragas(
    config: EvalConfig,
    outputs: list[PerRowOutput],
) -> dict[str, dict[str, float]]:
    """Group outputs by model and score each group via Ragas. Returns
    {model_name: {metric: score}}.

    Ragas needs a judge model (typically OpenAI). If credentials are missing
    the score step returns an empty dict and the runner reports rows + cost
    only — that is intentional, so the harness still produces a useful artifact
    on machines without judge-model access.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas import metrics as ragas_metrics  # type: ignore[import-not-found]
    except ImportError:
        return {}

    metric_map = {
        name: getattr(ragas_metrics, name)
        for name in config.metrics.ragas
        if hasattr(ragas_metrics, name)
    }
    if not metric_map:
        return {}

    grouped: dict[str, list[PerRowOutput]] = {}
    for o in outputs:
        grouped.setdefault(o.model_name, []).append(o)

    scores: dict[str, dict[str, float]] = {}
    for model_name, group in grouped.items():
        successful = [o for o in group if not o.error]
        if not successful:
            continue
        ds = Dataset.from_dict(
            {
                "question": [o.question for o in successful],
                "answer": [o.answer for o in successful],
                "contexts": [o.contexts for o in successful],
                "ground_truth": [o.ground_truth for o in successful],
            }
        )
        try:
            result = evaluate(ds, metrics=list(metric_map.values()))
        except Exception as e:  # noqa: BLE001 — judge-model error shouldn't kill the run
            scores[model_name] = {"_error": str(e)}  # type: ignore[dict-item]
            continue
        scores[model_name] = {
            k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))
        }
    return scores
