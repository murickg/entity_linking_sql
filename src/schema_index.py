import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.data_loader import load_ddl


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
    description: str = ""
    search_tokens: list[str] = field(default_factory=list)


@dataclass
class SchemaIndex:
    """BM25-based search index for a single database's schema."""

    db_name: str
    platform: str = "sqlite"
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
            # Add description tokens for Snowflake tables
            if info.description:
                tokens.extend(tokenize(info.description))
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
            # Show top-level columns only (no dot-paths) for preview
            col_names = [c[0] for c in info.columns if "." not in c[0]][:15]
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


def build_index(db_name: str, platform: str = "sqlite") -> SchemaIndex:
    """Build a SchemaIndex for a database by parsing its DDL.csv."""
    ddl_data = load_ddl(db_name, platform=platform)
    index = SchemaIndex(db_name=db_name, platform=platform)
    for table_name, table_data in ddl_data.items():
        index.tables[table_name] = TableInfo(
            name=table_name,
            ddl=table_data["ddl"],
            columns=table_data["columns"],
            description=table_data.get("description", ""),
        )
    index._build_table_index()
    return index


# Cache of built indices: key = (db_name, platform)
_index_cache: dict[tuple[str, str], SchemaIndex] = {}


def get_index(db_name: str, platform: str = "sqlite") -> SchemaIndex:
    """Get or build a cached SchemaIndex for a database."""
    key = (db_name, platform)
    if key not in _index_cache:
        _index_cache[key] = build_index(db_name, platform)
    return _index_cache[key]
