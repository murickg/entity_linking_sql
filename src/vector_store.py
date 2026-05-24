import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, VS_INITIAL_TOP_N, VS_RETRIEVE_TOP_M
from src.sqlite_executor import SQLiteExecutor
from src.column_describer import generate_all_descriptions


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25: split camelCase, underscores, non-alpha."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-/.]", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 0]


@dataclass
class ColumnDocument:
    table_name: str
    column_name: str
    col_type: str
    sample_values: list[str] = field(default_factory=list)
    description: str = ""
    # Profiling stats (from AT&T paper)
    distinct_count: int | None = None
    null_count: int | None = None
    total_count: int | None = None
    min_value: str | None = None
    max_value: str | None = None

    def to_mschema(self) -> str:
        """Format as M-Schema string for embedding and display."""
        nested = "." in self.column_name
        col_label = self.column_name
        parts = [f"Table: {self.table_name}, Column: {col_label}"]
        if nested:
            top = self.column_name.split(".")[0]
            parts.append(f"Nested field inside {top} STRUCT/RECORD")
        if self.col_type:
            parts.append(f"Type: {self.col_type}")
        if self.description:
            parts.append(f"Description: {self.description}")
        # Profiling stats — enrich embedding
        if self.distinct_count is not None and self.total_count:
            null_pct = round(100 * (self.null_count or 0) / self.total_count) if self.total_count else 0
            parts.append(f"Stats: {self.distinct_count} distinct / {self.total_count} rows, {null_pct}% null")
        if self.min_value is not None and self.max_value is not None:
            parts.append(f"Range: [{self.min_value} .. {self.max_value}]")
        if self.sample_values:
            vals = ", ".join(f'"{v}"' for v in self.sample_values[:5])
            parts.append(f"Values: [{vals}]")
        return ", ".join(parts)


class VectorStore:
    """Embedding-based vector store for column-level semantic retrieval (E_VS)."""

    def __init__(self, db_name: str, model_name: str = EMBEDDING_MODEL):
        self.db_name = db_name
        self.model_name = model_name
        self.documents: list[ColumnDocument] = []
        self.embeddings: np.ndarray | None = None
        self._model = None
        self._bm25: BM25Okapi | None = None
        self._tokenized_docs: list[list[str]] = []
        self._excluded: set[int] = set()  # indices to exclude from retrieval

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(
        self,
        ddl_data: dict[str, dict],
        sqlite_path: Path | None = None,
        external_samples: dict[str, dict[str, list]] | None = None,
    ) -> None:
        """Build column documents from DDL metadata + profiling stats + sample values.

        Args:
            ddl_data: table_name -> {columns, description, ...}
            sqlite_path: for SQLite, used to query sample values + profiling live
            external_samples: for Snowflake/BQ, pre-loaded {table: {col: [values]}}
        """
        executor = SQLiteExecutor(sqlite_path) if sqlite_path else None
        self.documents = []

        # Pre-fetch row counts per table (SQLite only)
        table_counts: dict[str, int] = {}
        if executor is not None:
            for table_name in ddl_data:
                result = executor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                try:
                    lines = result.strip().split("\n")
                    if len(lines) >= 3:
                        table_counts[table_name] = int(lines[-1].strip().split("|")[0].strip())
                except (ValueError, IndexError):
                    pass

        for table_name, table_data in ddl_data.items():
            columns = table_data.get("columns", [])
            total_count = table_counts.get(table_name)

            for col_name, col_type in columns:
                is_nested = "." in col_name

                # For SQLite (no nested types), keep old behavior — skip dot-paths
                if is_nested and executor is not None:
                    continue

                # Sample values: live for SQLite, JSON-derived for others
                if executor is not None:
                    sample_values = executor.get_sample_values(table_name, col_name, limit=10)
                elif external_samples and table_name in external_samples:
                    # Try direct lookup first (e.g. "trafficSource.source")
                    raw = external_samples[table_name].get(col_name, [])
                    if not raw and is_nested:
                        # Walk nested dict path: ["trafficSource", "source"]
                        parts = col_name.split(".")
                        cur = external_samples[table_name].get(parts[0], [])
                        # cur is list of values (dicts for nested STRUCTs)
                        nested_vals = []
                        for v in cur:
                            if isinstance(v, dict):
                                node = v
                                for p in parts[1:]:
                                    if isinstance(node, dict):
                                        node = node.get(p)
                                    else:
                                        node = None
                                        break
                                if node is not None:
                                    nested_vals.append(node)
                        raw = nested_vals
                    sample_values = [str(v) for v in raw if v is not None][:10]
                else:
                    sample_values = []

                # Profiling stats — SQLite-only (expensive on Snowflake/BQ)
                distinct_count = None
                null_count = None
                min_value = None
                max_value = None

                if executor is not None and not is_nested:
                    profile_sql = (
                        f'SELECT COUNT(DISTINCT "{col_name}"), '
                        f'SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END), '
                        f'MIN("{col_name}"), MAX("{col_name}") '
                        f'FROM "{table_name}"'
                    )
                    profile_result = executor.execute(profile_sql)
                    if "[ERROR" not in profile_result:
                        try:
                            lines = profile_result.strip().split("\n")
                            if len(lines) >= 3:
                                vals = [v.strip() for v in lines[-1].split("|")]
                                if len(vals) >= 4:
                                    distinct_count = int(vals[0]) if vals[0] else None
                                    null_count = int(vals[1]) if vals[1] else None
                                    min_val = vals[2].strip()
                                    max_val = vals[3].strip()
                                    min_value = min_val[:40] if min_val else None
                                    max_value = max_val[:40] if max_val else None
                        except (ValueError, IndexError):
                            pass

                doc = ColumnDocument(
                    table_name=table_name,
                    column_name=col_name,
                    col_type=col_type,
                    sample_values=sample_values,
                    description=table_data.get("description", ""),
                    distinct_count=distinct_count,
                    null_count=null_count,
                    total_count=total_count,
                    min_value=min_value,
                    max_value=max_value,
                )
                self.documents.append(doc)

        # Generate LLM descriptions (cached per db)
        self._apply_llm_descriptions()

        # Build embeddings
        self._build_embeddings()

        # Build BM25 index (lexical, complements semantic search)
        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build BM25 index over column docs for lexical retrieval."""
        if not self.documents:
            self._bm25 = None
            return
        self._tokenized_docs = []
        for doc in self.documents:
            # Tokenize table + column + description + sample values
            text = f"{doc.table_name} {doc.column_name} {doc.description} {' '.join(doc.sample_values[:3])}"
            self._tokenized_docs.append(_tokenize(text))
        self._bm25 = BM25Okapi(self._tokenized_docs)

    def _apply_llm_descriptions(self) -> None:
        """Generate and apply LLM descriptions to column documents."""
        if not self.documents:
            return
        descriptions = generate_all_descriptions(self.db_name, self.documents)
        applied = 0
        for doc in self.documents:
            key = f"{doc.table_name}.{doc.column_name}"
            if key in descriptions:
                doc.description = descriptions[key]
                applied += 1
        if applied:
            print(f"  Applied {applied}/{len(self.documents)} LLM descriptions")

    def _build_embeddings(self) -> None:
        """Encode all column documents and cache."""
        cache_path = self._cache_path()

        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            cached_texts = data["texts"].tolist()
            current_texts = [doc.to_mschema() for doc in self.documents]
            # Validate cache matches current documents
            if cached_texts == current_texts:
                self.embeddings = data["embeddings"]
                return

        # Encode
        model = self._get_model()
        texts = [doc.to_mschema() for doc in self.documents]
        if not texts:
            self.embeddings = np.array([])
            return

        print(f"  Encoding {len(texts)} column docs with {self.model_name}...")
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=self.embeddings, texts=np.array(texts, dtype=object))

    def _cache_path(self) -> Path:
        return EMBEDDING_CACHE_DIR / f"{self.db_name}.npz"

    def _vector_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([])
        model = self._get_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        return np.dot(self.embeddings, query_emb.T).flatten()

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None or not self._tokenized_docs:
            return np.array([])
        toks = _tokenize(query)
        if not toks:
            return np.zeros(len(self.documents))
        scores = self._bm25.get_scores(toks)
        # Normalize to 0..1 for fusion with cosine
        max_s = scores.max() if len(scores) else 0
        if max_s > 0:
            scores = scores / max_s
        return scores

    def _hybrid_scores(self, query: str, alpha: float = 0.7) -> np.ndarray:
        """Weighted fusion: alpha * vector + (1-alpha) * bm25.

        alpha=0.7 → favor semantic, but lexical helps with exact-match identifiers.
        """
        v = self._vector_scores(query)
        if len(v) == 0:
            return np.array([])
        b = self._bm25_scores(query)
        if len(b) == 0:
            return v
        return alpha * v + (1 - alpha) * b

    def retrieve(self, query: str, top_m: int = VS_RETRIEVE_TOP_M) -> list[ColumnDocument]:
        """Hybrid (vector + BM25) search: return top-m most similar columns."""
        scores = self._hybrid_scores(query)
        if len(scores) == 0:
            return []
        # Exclude already-linked columns
        for idx in self._excluded:
            if idx < len(scores):
                scores[idx] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_m]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

    def retrieve_initial(self, query: str, top_n: int = VS_INITIAL_TOP_N) -> list[ColumnDocument]:
        """Broad initial retrieval to seed S_linked (hybrid)."""
        scores = self._hybrid_scores(query)
        if len(scores) == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        for idx in top_indices:
            if scores[idx] > 0:
                self._excluded.add(int(idx))
        return results

    def mark_excluded(self, table_name: str, column_name: str) -> None:
        """Mark a column as already linked (exclude from future retrieval)."""
        for i, doc in enumerate(self.documents):
            if doc.table_name.lower() == table_name.lower() and doc.column_name.lower() == column_name.lower():
                self._excluded.add(i)
                break

    def reset_excluded(self) -> None:
        """Reset exclusion set (for new query)."""
        self._excluded.clear()


# Cache of built vector stores
_vs_cache: dict[str, VectorStore] = {}


def get_vector_store(
    db_name: str,
    ddl_data: dict,
    sqlite_path: Path | None = None,
    external_samples: dict | None = None,
) -> VectorStore:
    """Get or build a cached VectorStore."""
    if db_name not in _vs_cache:
        vs = VectorStore(db_name=db_name)
        vs.build(ddl_data, sqlite_path=sqlite_path, external_samples=external_samples)
        _vs_cache[db_name] = vs
    else:
        _vs_cache[db_name].reset_excluded()
    return _vs_cache[db_name]
