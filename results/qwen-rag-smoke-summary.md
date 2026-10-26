# Qwen 2.5 3B: closed-book versus BM25 RAG

Run date: 2026-10-27  
Model: `qwen2.5:3b` (Q4_K_M through Ollama 0.40.11)  
Decoding: temperature 0, maximum 160 output tokens  
Dataset: 21 questions; seven facts represented in English, Russian, and Kazakh  
Retrieval: language-filtered BM25, top 2 passages from 18 dated official-source excerpts

| Condition | Answer-key hit | Retrieval recall@2 | Correct gold citation | Failures | Mean latency |
|---|---:|---:|---:|---:|---:|
| Closed-book | 33.3% | — | — | 0% | 14.15 s |
| BM25 RAG | 81.0% | 100.0% | 81.0% | 0% | 9.60 s |

| Condition | English | Russian | Kazakh |
|---|---:|---:|---:|
| Closed-book | 28.6% | 71.4% | 0.0% |
| BM25 RAG | 100.0% | 85.7% | 57.1% |

The result supports a narrow engineering claim: on this small factual dataset, retrieval improved accepted-answer coverage for the same local model. It does not establish general RAG quality. The benchmark is intentionally transparent about three limitations: only seven independent facts are tested; accepted-answer substring matching can miss semantically correct paraphrases; and the source excerpts are curated snapshots rather than raw pages indexed end to end. The most informative failure is the Kazakh gap: retrieval recall remained 100%, but generation accuracy reached only 57.1%, so retrieval alone did not remove the model's language weakness.
