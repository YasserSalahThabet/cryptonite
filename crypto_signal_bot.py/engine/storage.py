# engine/storage.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "cryptonite.db"

@dataclass
class SignalRecord:
    ts: str
    symbol: str
    timeframe: str
    side: str
    price: float
    rsi: float
    ema_fast: float
    ema_slow: float
    confidence: float
    details: str

def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        side TEXT NOT NULL,
        price REAL NOT NULL,
        rsi REAL NOT NULL,
        ema_fast REAL NOT NULL,
        ema_slow REAL NOT NULL,
        confidence REAL NOT NULL,
        details TEXT NOT NULL
    );
    """
)

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
            ON signals(symbol, ts);
            """
        )

def insert_signal(rec: SignalRecord, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO signals
            (ts, symbol, timeframe, side, price, rsi, ema_fast, ema_slow, confidence, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                rec.ts,
                rec.symbol,
                rec.timeframe,
                rec.side,
                rec.price,
                rec.rsi,
                rec.ema_fast,
                rec.ema_slow,
                rec.confidence,
                rec.details,
            ),
        )

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
