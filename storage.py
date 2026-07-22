"""Durable storage for the chunk corpus behind BM25 sparse retrieval.

Deliberately free of Streamlit imports: this is plain data access, and keeping
it that way makes it usable outside a running app (scripts, tests, the eval
harness) and keeps every "what should the user see when this fails" decision in
app.py, where the answer depends on the UI context of the call.
"""

import os
import sqlite3

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "corpus.db")


def _connect():
    # A connection per call rather than one shared module-level handle:
    # Streamlit serves concurrent sessions as threads in a single process, and
    # sqlite3 connections are not safe to share across them.
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                namespace TEXT NOT NULL,
                idx       INTEGER NOT NULL,
                content   TEXT NOT NULL,
                PRIMARY KEY (namespace, idx)
            )
        """)


init_db()


def save_corpus(namespace, chunks):
    """Persist `chunks` for `namespace`, replacing anything stored previously.

    Delete-then-insert rather than upsert: re-indexing the same namespace with
    fewer chunks would otherwise leave the previous run's trailing rows behind,
    and those stale rows would shift nothing but still surface as phantom
    chunks. Both statements share one transaction, so a failure partway leaves
    the old corpus intact instead of half-deleted.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE namespace = ?", (namespace,))
        conn.executemany(
            "INSERT INTO chunks (namespace, idx, content) VALUES (?, ?, ?)",
            [(namespace, i, chunk) for i, chunk in enumerate(chunks)],
        )


def load_corpus(namespace):
    """Return the persisted chunks for `namespace`, or None if there are none.

    Order is load-bearing, hence ORDER BY idx: Pinecone stores each vector's
    position in the corpus list as metadata, and dense hits are resolved back to
    text by indexing into this list. A corpus rebuilt in a different order would
    silently answer with the wrong chunks.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE namespace = ? ORDER BY idx ASC",
            (namespace,),
        ).fetchall()
    return [row[0] for row in rows] if rows else None


def delete_corpus(namespace):
    """Drop a namespace's chunks. Safe to call on one that was never stored."""
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE namespace = ?", (namespace,))
