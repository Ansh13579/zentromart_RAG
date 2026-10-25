"""Embedding and FAISS retrieval for the ZentroMart knowledge base."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import warnings

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if __package__:
    from .config import (
        RERANKER_BATCH_SIZE,
        RERANKER_CANDIDATE_COUNT,
        RERANKER_ENABLED,
        RERANKER_MAX_LENGTH,
        RERANKER_MODEL,
    )
else:
    from config import (
        RERANKER_BATCH_SIZE,
        RERANKER_CANDIDATE_COUNT,
        RERANKER_ENABLED,
        RERANKER_MAX_LENGTH,
        RERANKER_MODEL,
    )


DEFAULT_CHUNKS_PATH = (
    Path(__file__).resolve().parents[1] / "indexes" / "chunks.json"
)
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "indexes"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u0900-\u097f]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "the",
    "to", "was", "what", "when", "where", "which", "with", "you", "your",
    "hai", "hain", "ho", "ka", "ki", "ko", "kya", "mein", "mujhe", "pe",
    "क्या", "कब", "कहाँ", "कैसे", "कितना", "कितने", "की", "को", "के", "है",
    "हैं", "मुझे", "में", "मैं", "और", "पर", "से", "तो",
}
_QUERY_ALIASES = {
    "wapas": {"return", "returns"},
    "वापस": {"return", "returns"},
    "लौटाना": {"return", "returns"},
    "लौटाऊँ": {"return", "returns"},
    "रिटर्न": {"return", "returns"},
    "refund": {"refund", "refunded", "refunds"},
    "रिफंड": {"refund", "refunded", "refunds"},
    "वापसी": {"refund", "return"},
    "din": {"day", "days", "window", "timeline"},
    "दिन": {"day", "days", "window", "timeline"},
    "कितने": {"how", "many", "days", "window"},
    "khola": {"open", "opened", "seal", "sealed"},
    "खोला": {"open", "opened", "seal", "sealed"},
    "damage": {"damaged", "defective", "broken"},
    "damaged": {"damage", "defective", "broken"},
    "फटा": {"torn", "damaged", "damage"},
    "टूटा": {"broken", "damaged", "damage"},
    "free": {"free", "fee", "fees", "charge", "charges"},
    "मुफ्त": {"free", "fee", "fees", "charge"},
    "शुल्क": {"fee", "fees", "charge", "charges"},
    "delivery": {"delivery", "shipment", "shipping"},
    "डिलीवरी": {"delivery", "shipment", "shipping"},
    "order": {"order", "order_id", "shipment"},
    "ऑर्डर": {"order", "order_id", "shipment"},
    "warranty": {"warranty", "assured", "guarantee"},
    "वारंटी": {"warranty", "assured", "guarantee"},
}


class Retriever:
    """Build, persist and query a normalized inner-product FAISS index."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        model: SentenceTransformer | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or SentenceTransformer(
            model_name,
            token=os.getenv("HF_TOKEN"),
        )
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self.reranker = reranker
        self._reranker_attempted = reranker is not None

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Create an in-memory index from knowledge-base chunk dictionaries."""

        if not chunks:
            raise ValueError("Cannot build a retriever from zero chunks.")

        texts = [self._chunk_text(chunk) for chunk in chunks]
        embeddings = self._encode(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = [dict(chunk) for chunk in chunks]

    def build_from_json(self, chunks_path: Path) -> None:
        """Load chunk dictionaries from JSON and build the in-memory index."""

        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunk file was not found: {chunks_path}")
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Chunk JSON must contain a list of objects.")
        self.build(data)

    def save(self, index_dir: Path) -> None:
        """Save the FAISS index and its metadata without serializing Python objects."""

        self._require_index()
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "vector_index.faiss"))
        (index_dir / "retriever_metadata.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "chunk_count": len(self.chunks),
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, index_dir: Path) -> None:
        """Load a previously saved FAISS index and its chunk metadata."""

        index_path = index_dir / "vector_index.faiss"
        metadata_path = index_dir / "retriever_metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Both {index_path.name} and {metadata_path.name} are required."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks")
        if not isinstance(chunks, list) or not all(
            isinstance(item, dict) for item in chunks
        ):
            raise ValueError("Retriever metadata contains invalid chunks.")

        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks):
            raise ValueError(
                "Index and metadata are inconsistent: "
                f"{index.ntotal} vectors for {len(chunks)} chunks."
            )
        self.index = index
        self.chunks = chunks
        self.model_name = metadata.get("model_name", self.model_name)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve a candidate pool, then return its best reranked chunks.

        The embedding model is English-focused, while the practice questions
        include Hindi and Hinglish. BM25, bilingual query aliases, and
        policy-precedence signals help find candidates; an optional multilingual
        cross-encoder then scores query/passage pairs for final ordering.
        """

        self._require_index()
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self._encode([query])
        result_count = min(top_k, len(self.chunks))
        candidate_count = min(
            max(top_k, RERANKER_CANDIDATE_COUNT, top_k * 5),
            len(self.chunks),
        )
        scores, positions = self.index.search(query_embedding, candidate_count)
        semantic_scores = {
            int(position): float(score)
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        }
        lexical_scores = self._bm25_scores(query)
        max_lexical = max(lexical_scores.values(), default=0.0)
        ranked: list[tuple[float, float, int]] = []
        for position, score in semantic_scores.items():
            semantic = (score + 1.0) / 2.0
            lexical = lexical_scores.get(position, 0.0)
            lexical_normalized = lexical / max_lexical if max_lexical > 0 else 0.0
            # Keep semantic meaning as the main signal, while allowing exact
            # product/policy terms to recover cross-language vector misses.
            combined = 0.68 * semantic + 0.32 * lexical_normalized
            chunk = self.chunks[position]
            status = str(chunk.get("status", "")).casefold()
            doc_type = str(chunk.get("doc_type", "")).casefold()
            if status == "superseded":
                combined -= 0.12
            elif status == "current" and doc_type == "policy":
                combined += 0.035
            elif status == "current" and doc_type == "agent_guide":
                combined += 0.02
            ranked.append((combined, score, position))

        ranked.sort(key=lambda item: item[0], reverse=True)
        candidates = [
            (combined, score, position)
            for combined, score, position in ranked[:candidate_count]
        ]
        reranker = self._get_reranker()
        if reranker is not None and candidates:
            try:
                pairs = [
                    (
                        query,
                        self._reranker_passage(self.chunks[position]),
                    )
                    for _, _, position in candidates
                ]
                reranker_scores = np.asarray(
                    reranker.predict(
                        pairs,
                        batch_size=RERANKER_BATCH_SIZE,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if len(reranker_scores) != len(candidates):
                    raise ValueError(
                        "Reranker returned a different number of scores than candidates."
                    )
                rescored: list[tuple[float, float, float, int]] = []
                for (hybrid_score, semantic_score, position), raw_score in zip(
                    candidates,
                    reranker_scores,
                ):
                    relevance = self._sigmoid(float(raw_score))
                    final_score = 0.84 * relevance + 0.16 * hybrid_score
                    chunk = self.chunks[position]
                    status = str(chunk.get("status", "")).casefold()
                    doc_type = str(chunk.get("doc_type", "")).casefold()
                    if status == "superseded":
                        final_score -= 0.10
                    elif status == "current" and doc_type == "policy":
                        final_score += 0.025
                    rescored.append(
                        (final_score, semantic_score, float(raw_score), position)
                    )
                rescored.sort(key=lambda item: item[0], reverse=True)
                ranked_results = [
                    (final, semantic, raw, position)
                    for final, semantic, raw, position in rescored[:result_count]
                ]
            except Exception as error:
                self._disable_reranker(
                    f"Reranker scoring failed; using hybrid retrieval for this run: {error}"
                )
                ranked_results = [
                    (combined, semantic, float("nan"), position)
                    for combined, semantic, position in candidates[:result_count]
                ]
        else:
            ranked_results = [
                (combined, semantic, float("nan"), position)
                for combined, semantic, position in candidates[:result_count]
            ]

        results: list[dict[str, Any]] = []
        for rank_score, score, reranker_score, position in ranked_results:
            if position < 0:
                continue
            result = dict(self.chunks[position])
            result["score"] = float(score)
            result["hybrid_score"] = float(
                next(
                    (hybrid for hybrid, _, candidate_position in candidates if candidate_position == position),
                    rank_score,
                )
            )
            result["rank_score"] = float(rank_score)
            if math.isfinite(reranker_score):
                result["reranker_score"] = float(reranker_score)
            results.append(result)
        return results

    def _get_reranker(self) -> Any | None:
        """Load the small multilingual cross-encoder lazily on CPU."""

        if not RERANKER_ENABLED:
            return None
        if self._reranker_attempted:
            return self.reranker
        self._reranker_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                RERANKER_MODEL,
                device="cpu",
                max_length=RERANKER_MAX_LENGTH,
            )
        except Exception as error:
            self._disable_reranker(
                f"Could not load reranker {RERANKER_MODEL!r}; using hybrid retrieval: {error}"
            )
        return self.reranker

    @staticmethod
    def _reranker_passage(chunk: dict[str, Any]) -> str:
        """Provide policy metadata alongside the text being relevance-scored."""

        return "\n".join(
            (
                f"Section: {chunk.get('section_id', '')}",
                f"Title: {chunk.get('title', '')}",
                f"Document type: {chunk.get('doc_type', '')}",
                f"Status: {chunk.get('status', '')}",
                str(chunk.get("text", "")),
            )
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Map an unbounded cross-encoder logit into a stable 0-1 score."""

        clipped = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    def _disable_reranker(self, message: str) -> None:
        self.reranker = None
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _bm25_scores(self, query: str) -> dict[int, float]:
        """Score chunks lexically, expanding common Hindi/Hinglish policy terms."""

        query_terms = self._query_terms(query)
        if not query_terms:
            return {}

        documents = [
            self._query_terms(
                " ".join(
                    str(chunk.get(key, ""))
                    for key in ("section_id", "title", "doc_type", "status", "text")
                )
            )
            for chunk in self.chunks
        ]
        lengths = [len(document) for document in documents]
        average_length = sum(lengths) / len(lengths) if lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))

        scores: dict[int, float] = {}
        total_documents = len(documents)
        k1 = 1.5
        b = 0.75
        for position, document in enumerate(documents):
            term_counts = Counter(document)
            length_factor = 1.0 - b + b * lengths[position] / max(average_length, 1.0)
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (total_documents - df + 0.5) / (df + 0.5)
                )
                score += inverse_document_frequency * (
                    frequency * (k1 + 1.0)
                    / (frequency + k1 * length_factor)
                )
            if score:
                scores[position] = score
        return scores

    @staticmethod
    def _query_terms(text: str) -> list[str]:
        """Tokenize text and add small domain-specific bilingual expansions."""

        terms = [
            match.group(0).casefold()
            for match in _TOKEN_PATTERN.finditer(text)
            if match.group(0).casefold() not in _STOP_WORDS
        ]
        expanded = list(terms)
        for term in terms:
            expanded.extend(_QUERY_ALIASES.get(term, ()))
        return expanded

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        faiss.normalize_L2(values)
        return values

    @staticmethod
    def _chunk_text(chunk: dict[str, Any]) -> str:
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every chunk must contain non-empty text.")
        return text

    def _require_index(self) -> None:
        if self.index is None or not self.chunks:
            raise RuntimeError("Build or load the retriever before searching.")


def build_default_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Retriever:
    """Build and save the default knowledge-base index."""

    retriever = Retriever(model_name=model_name)
    retriever.build_from_json(chunks_path)
    retriever.save(index_dir)
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ZentroMart FAISS knowledge-base index."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query", help="Optional query to test after indexing")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = build_default_index(args.chunks, args.index_dir, args.model)
    print(f"Indexed {len(retriever.chunks)} chunks in {args.index_dir}")
    if args.query:
        print(json.dumps(retriever.search(args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
