# KZ RAG Eval

KZ RAG Eval compares how the same multilingual model answers Kazakhstan public-information questions with and without retrieved official evidence. The evaluation dataset contains parallel Kazakh, Russian, and English questions grounded in eGov Kazakhstan pages.

The project is adapted from Paiteq's MIT-licensed [`ai-eval-harness`](https://github.com/paiteq/ai-eval-harness). The Ollama integration, Kazakhstan dataset, baseline/RAG experiment, and subsequent analysis are maintained here.

## What the experiment tests

Both conditions use the same `qwen2.5:3b` model and deterministic decoding. The 3B model is selected so the benchmark can run on an 8 GB Apple-silicon Mac:

- **Baseline:** the model receives only the question.
- **RAG:** BM25 retrieves two passages from a separate multilingual corpus;
  the model receives those passages and must cite their document IDs.

The current benchmark contains 21 cases covering seven independent facts in parallel Kazakh, Russian, and English. It provides an end-to-end measured comparison, while the planned expanded release will cover 50 underlying questions in three languages (150 cases).

## First local result

The first deterministic run with `qwen2.5:3b` completed without failures. RAG increased exact accepted-answer coverage from 33.3% to 81.0%, while BM25 retrieved a gold source for all 21 questions. The language split is important: RAG scored 100.0% in English, 85.7% in Russian, and 57.1% in Kazakh. See the [run summary](results/qwen-rag-smoke-summary.md) for the protocol, latency, and limitations.

## Run locally

Requirements: Python 3.10+, [Ollama](https://ollama.com/), and the Qwen model.

```bash
ollama pull qwen2.5:3b
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

kz-rag-eval run configs/qwen-rag-smoke.yaml --no-score --out runs/
```

Run the checks with:

```bash
pytest -q
```

## Current limitations

- Source excerpts are curated, dated snapshots of linked official pages; retrieval itself is automatic.
- The evaluation dataset has 21 cases and is too small to support a broad claim about multilingual RAG performance.
- Answer correctness uses transparent accepted-answer matching, which is appropriate for these short factual questions but not a substitute for human review on open-ended questions.
- Source-page contents can change; benchmark releases should preserve dated snapshots.

Each run writes a machine-readable JSON file and a Markdown report containing aggregate metrics, retrieved source IDs, and every answer. The reported metrics are answer-key hit rate, retrieval recall@k, correct-citation rate, failures, latency, and token use.

## License and attribution

MIT. See [`LICENSE`](LICENSE). The original harness was created by Paiteq; see the upstream repository for its authors and history.
