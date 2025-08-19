# src/memory/store.py
from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging import get_logger

log = get_logger("memory.store")


@dataclass
class MemoryItem:
    rowid: int
    user_id: str
    kind: str
    content: str
    created_at: int
    ttl_days: int
    base_score: float


class MemoryStore:
    def __init__(
        self,
        db_path: str,
        ttl_days: int = 30,
        half_life_days: int = 14,
        max_items: int = 500,
    ):
        self.db_path = db_path
        self.ttl_days = int(ttl_days)
        self.half_life_days = int(half_life_days)
        self.max_items = int(max_items)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    user_id   TEXT    NOT NULL,
                    kind      TEXT    NOT NULL DEFAULT 'fact',
                    content   TEXT    NOT NULL,
                    created_at INTEGER NOT NULL,
                    ttl_days  INTEGER NOT NULL DEFAULT 30,
                    base_score REAL   NOT NULL DEFAULT 1.0
                )
            """)
            # Índice FTS5 para búsquedas rápidas por texto
            cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(content)""")
            # Triggers de sync entre tabla base y FTS
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                  INSERT INTO mem_fts(rowid, content) VALUES (last_insert_rowid(), NEW.content);
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                  DELETE FROM mem_fts WHERE rowid = old.rowid;
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                  UPDATE mem_fts SET content = NEW.content WHERE rowid = NEW.rowid;
                END
            """)
            con.commit()
        finally:
            con.close()

    def _now(self) -> int:
        return int(time.time())

    def _days_ago(self, ts: int) -> float:
        return max(0.0, (self._now() - int(ts)) / 86400.0)

    def _decayed(self, base_score: float, age_days: float) -> float:
        # score = base * 2^(-age/half_life)
        if self.half_life_days <= 0:
            return base_score
        return float(base_score) * pow(2.0, -age_days / float(self.half_life_days))

    # ------------------- API pública -------------------

    def upsert(
        self,
        user_id: str,
        content: str,
        kind: str = "fact",
        ttl_days: Optional[int] = None,
        base_score: float = 1.0,
    ) -> int:
        """Inserta un recuerdo. Retorna rowid (o 0 si content vacío)."""
        content = (content or "").strip()
        if not content:
            return 0
        ttl = int(ttl_days if ttl_days is not None else self.ttl_days)

        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO memories(user_id, kind, content, created_at, ttl_days, base_score) VALUES (?,?,?,?,?,?)",
                (user_id, kind, content, self._now(), ttl, float(base_score)),
            )
            con.commit()
            rid = cur.lastrowid
        finally:
            con.close()

        # GC suave por usuario
        self._gc_user(user_id)
        return int(rid)

    def _gc_user(self, user_id: str):
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            # 1) borrar vencidos por TTL
            cur.execute("""
                DELETE FROM memories
                WHERE user_id = ?
                  AND (strftime('%s','now') - created_at) > (ttl_days * 86400)
            """, (user_id,))
            con.commit()
            # 2) limitar cantidad total
            cur.execute("SELECT rowid FROM memories WHERE user_id=? ORDER BY created_at DESC", (user_id,))
            rows = [r[0] for r in cur.fetchall()]
            if len(rows) > self.max_items:
                to_delete = rows[self.max_items:]
                q = "DELETE FROM memories WHERE rowid IN ({})".format(",".join("?" * len(to_delete)))
                cur.execute(q, to_delete)
                con.commit()
        finally:
            con.close()

    def query(self, user_id: str, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        Devuelve [(MemoryItem, score_final)] ordenado por score_final
        (BM25 de FTS5 + decaimiento exponencial).
        """
        if not (query or "").strip():
            return []

        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            cur.execute("""
                SELECT m.rowid AS rowid, m.user_id, m.kind, m.content, m.created_at, m.ttl_days, m.base_score,
                       bm25(mem_fts) AS bm25_score
                FROM mem_fts
                JOIN memories m ON m.rowid = mem_fts.rowid
                WHERE m.user_id = ? AND mem_fts MATCH ?
                ORDER BY bm25(mem_fts) ASC
                LIMIT 100
            """, (user_id, query))
            rows = cur.fetchall()
        finally:
            con.close()

        if not rows:
            return []

        bm25_vals = [float(r["bm25_score"]) for r in rows]
        mx = max(bm25_vals) if bm25_vals else 1.0
        mn = min(bm25_vals) if bm25_vals else 0.0
        rng = max(1e-9, (mx - mn))

        items: List[Tuple[MemoryItem, float]] = []
        for r in rows:
            age = self._days_ago(int(r["created_at"]))
            decay = self._decayed(float(r["base_score"]), age)
            bm25_norm = 1.0 - ((float(r["bm25_score"]) - mn) / rng)  # 1.0 es mejor
            final = 0.7 * bm25_norm + 0.3 * decay
            items.append((
                MemoryItem(
                    rowid=int(r["rowid"]),
                    user_id=r["user_id"],
                    kind=r["kind"],
                    content=r["content"],
                    created_at=int(r["created_at"]),
                    ttl_days=int(r["ttl_days"]),
                    base_score=float(r["base_score"]),
                ),
                float(final),
            ))

        items.sort(key=lambda t: t[1], reverse=True)
        return items[:top_k]

    def top_text_block(self, user_id: str, query: str, top_k: int = 5) -> str:
        """Bloque de texto listo para inyectar como [MEMORIA] en el prompt."""
        pairs = self.query(user_id, query, top_k=top_k)
        if not pairs:
            return ""
        lines = [f"- {it.content}" for it, _ in pairs]
        return "Recuerdos del usuario relevantes:\n" + "\n".join(lines)


# ------------- Singleton práctico -------------

_STORE: Optional[MemoryStore] = None

def init_memory(cfg: Dict) -> MemoryStore:
    """
    Espera un dict de configuración completo (el que retorna load_cfg()).
    Toma la sección cfg["memory"]. Si no existe, usa defaults razonables.
    """
    global _STORE
    if not isinstance(cfg, dict):
        log.warning(f"init_memory: se esperaba dict cfg, se recibió {type(cfg).__name__}. Uso defaults.")
        mcfg = {}
    else:
        mcfg = cfg.get("memory") or {}

    db_path = (mcfg.get("db_path") or "cache/memory.sqlite3")
    ttl_days = int(mcfg.get("ttl_days", 30))
    half_life = int(mcfg.get("half_life_days", 14))
    max_items = int(mcfg.get("max_items", 500))

    _STORE = MemoryStore(
        db_path,
        ttl_days=ttl_days,
        half_life_days=half_life,
        max_items=max_items,
    )
    log.info(f"MemoryStore listo (db={db_path}, ttl={ttl_days}d, half_life={half_life}d, max={max_items})")
    return _STORE

def get_store() -> MemoryStore:
    """Obtiene la instancia global inicializada. Llamar init_memory(cfg) antes."""
    if _STORE is None:
        raise RuntimeError("MemoryStore no inicializado. Llamá init_memory(cfg) durante el boot.")
    return _STORE
