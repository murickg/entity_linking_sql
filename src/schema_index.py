import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.data_loader import load_ddl, load_local_map


def tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on underscores, spaces, camelCase boundaries."""
    # Split camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace underscores and non-alphanumeric with spaces
    text = re.sub(r'[_\-/.]', ' ', text)
    tokens = text.lower().split()
    return [t for t in tokens if len(t) > 0]


@dataclass
class TableInfo:
    name: str
    ddl: str
    columns: list[tuple[str, str]]  # (name, type)
    search_tokens: list[str] = field(default_factory=list)


@dataclass
class SchemaIndex:
    """BM25-based search index for a single database's schema."""

    db_name: str
    tables: dict[str, TableInfo] = field(default_factory=dict)
    _table_bm25: BM25Okapi | None = None
    _table_names: list[str] = field(default_factory=list)
    _column_bm25: dict[str, BM25Okapi] = field(default_factory=dict)
    _column_names: dict[str, list[str]] = field(default_factory=dict)

    def _build_table_index(self):
        corpus = []
        self._table_names = []
        for name, info in self.tables.items():
            tokens = tokenize(name)
            for col_name, col_type in info.columns:
                tokens.extend(tokenize(col_name))
            info.search_tokens = tokens
            corpus.append(tokens)
            self._table_names.append(name)
        if corpus:
            self._table_bm25 = BM25Okapi(corpus)

    def _build_column_index(self, table_name: str):
        info = self.tables.get(table_name)
        if not info:
            return
        corpus = []
        names = []
        for col_name, col_type in info.columns:
            tokens = tokenize(col_name)
            if col_type:
                tokens.extend(tokenize(col_type))
            corpus.append(tokens)
            names.append(col_name)
        if corpus:
            self._column_bm25[table_name] = BM25Okapi(corpus)
            self._column_names[table_name] = names

    def search_tables(self, query: str, top_k: int = 10) -> list[dict]:
        """Search tables by query. Returns list of {name, score, columns_preview}."""
        if not self._table_bm25:
            return []
        tokens = tokenize(query)
        scores = self._table_bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            name = self._table_names[idx]
            info = self.tables[name]
            col_names = [c[0] for c in info.columns[:15]]
            results.append({
                "table_name": name,
                "score": round(float(score), 4),
                "columns": col_names,
            })
        return results

    def search_columns(self, table_name: str, query: str, top_k: int = 20) -> list[dict]:
        """Search columns within a table. Returns list of {name, type, score}."""
        if table_name not in self._column_bm25:
            self._build_column_index(table_name)
        bm25 = self._column_bm25.get(table_name)
        if not bm25:
            return []
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        info = self.tables[table_name]
        results = []
        for idx, score in ranked:
            col_name, col_type = info.columns[idx]
            results.append({
                "column_name": col_name,
                "type": col_type,
                "score": round(float(score), 4),
            })
        return results

    def get_table_schema(self, table_name: str) -> str | None:
        """Return the full DDL for a table."""
        info = self.tables.get(table_name)
        if not info:
            return None
        return info.ddl


def build_index(db_name: str) -> SchemaIndex:
    """Build a SchemaIndex for a SQLite database by parsing its DDL.csv."""
    ddl_data = load_ddl(db_name)
    index = SchemaIndex(db_name=db_name)
    for table_name, table_data in ddl_data.items():
        index.tables[table_name] = TableInfo(
            name=table_name,
            ddl=table_data["ddl"],
            columns=table_data["columns"],
        )
    index._build_table_index()
    return index


# Cache of built indices
_index_cache: dict[str, SchemaIndex] = {}


def get_index(db_name: str) -> SchemaIndex:
    """Get or build a cached SchemaIndex for a database."""
    if db_name not in _index_cache:
        _index_cache[db_name] = build_index(db_name)
    return _index_cache[db_name]


def resolve_db_name(instance_id: str, local_map: dict[str, str] | None = None) -> str | None:
    """Resolve instance_id to a database name."""
    if local_map is None:
        local_map = load_local_map()
    return local_map.get(instance_id)
