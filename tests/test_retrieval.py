from __future__ import annotations

from pathlib import Path

from ai_eval_harness.config import EvalConfig
from ai_eval_harness.loaders import load_rows
from ai_eval_harness.retrieval import BM25Retriever, load_corpus
from ai_eval_harness.runner import PerRowOutput, _cited_doc_ids, _score_deterministic

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "qwen-rag-smoke.yaml"


def test_bm25_retrieves_a_gold_source_for_every_question() -> None:
    cfg = EvalConfig.from_yaml(CONFIG)
    rows = load_rows(cfg.dataset)
    retriever = BM25Retriever(load_corpus(cfg.retrieval.corpus_source or ""))

    for row in rows:
        retrieved = retriever.retrieve(
            str(row["question"]), cfg.retrieval.top_k, language=str(row["language"])
        )
        retrieved_ids = {document.id for document in retrieved}
        assert retrieved_ids & set(row["source_ids"]), row["id"]
        assert all(document.language == row["language"] for document in retrieved)


def test_deterministic_metrics_measure_answers_retrieval_and_citations() -> None:
    outputs = [
        PerRowOutput(
            question="How many digits?",
            ground_truth="12 digits",
            contexts=[],
            answer="It contains 12 digits [iin-en].",
            model_name="rag",
            accepted_answers=["12 digits"],
            source_ids=["iin-en"],
            retrieved_doc_ids=["iin-en", "documents-en"],
        ),
        PerRowOutput(
            question="How long?",
            ground_truth="15 working days",
            contexts=[],
            answer="I do not know.",
            model_name="rag",
            accepted_answers=["15 working days"],
            source_ids=["documents-en"],
            retrieved_doc_ids=["iin-en"],
        ),
    ]

    scores = _score_deterministic(outputs)["rag"]
    assert scores["answer_key_hit_rate"] == 0.5
    assert scores["retrieval_recall_at_k"] == 0.5
    assert scores["correct_citation_rate"] == 0.5


def test_source_label_is_accepted_inside_citation_brackets() -> None:
    assert _cited_doc_ids("Twelve digits [Source: iin-en].") == {"iin-en"}
