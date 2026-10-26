"""Small, dependency-free BM25 retriever for reproducible local experiments."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    source_url: str = ""
    language: str = ""

    def prompt_text(self) -> str:
        source = f"\nSource: {self.source_url}" if self.source_url else ""
        title = f"{self.title}\n" if self.title else ""
        return f"[{self.id}]\n{title}{self.text}{source}"


def load_corpus(path: str | Path) -> list[Document]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    documents: list[Document] = []
    seen: set[str] = set()
    with corpus_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            doc_id = str(raw.get("id", "")).strip()
            text = str(raw.get("text", "")).strip()
            if not doc_id or not text:
                raise ValueError(f"{corpus_path}:{line_number}: document needs id and text")
            if doc_id in seen:
                raise ValueError(f"duplicate document id: {doc_id}")
            seen.add(doc_id)
            documents.append(
                Document(
                    id=doc_id,
                    text=text,
                    title=str(raw.get("title", "")),
                    source_url=str(raw.get("source_url", "")),
                    language=str(raw.get("language", "")),
                )
            )
    if not documents:
        raise ValueError(f"corpus is empty: {corpus_path}")
    return documents


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class BM25Retriever:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("at least one document is required")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._terms = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._avg_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            document_frequency.update(terms.keys())
        count = len(documents)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        language: str | None = None,
    ) -> list[Document]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = set(tokenize(query))
        scores: list[tuple[float, int]] = []
        for index, terms in enumerate(self._terms):
            if language and self.documents[index].language not in {"", language}:
                continue
            length_norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avg_length)
            score = sum(
                self._idf.get(term, 0.0) * terms[term] * (self.k1 + 1) / (terms[term] + length_norm)
                for term in query_terms
                if terms[term]
            )
            scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], self.documents[item[1]].id))
        return [self.documents[index] for _, index in scores[:top_k]]
