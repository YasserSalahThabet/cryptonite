#!/usr/bin/env python3
from __future__ import annotations

"""
Cryptonite – Signal Bot
- Coinbase via CCXT
- EMA(20/50) + RSI strategy
- Confidence scoring (rule-based)
- SQLite logging (data/cryptonite.db)
- Telegram alerts (.env via python-dotenv)
- Duplicate signal protection (dedup window)

Run:
  cd ~/Desktop/crypto_signal_bot.py
  python3 signal_bot.py
"""

import os
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import requests
import ccxt
import pandas as pd
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# -------------------- PATHS + ENV (.env) --------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load .env explicitly from project folder
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ✅ Correct env var names (YOUR CURRENT FILE USED THE TOKEN VALUE AS THE KEY — that breaks everything)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# "Pro" flags (UI/backend later, but bot is ready)
PRO_MODE = os.getenv("PRO_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
# Optional: comma-separated list of allowed chat IDs (future use)
PRO_IDS_RAW = os.getenv("PRO_IDS", "").strip()
PRO_IDS = {x.strip() for x in PRO_IDS_RAW.split(",") if x.strip()}

# One-time welcome message toggle
SEND_WELCOME_ON_START = os.getenv("SEND_WELCOME_ON_START", "0").strip().lower() in ("1", "true", "yes", "on")
PREFERRED_PLATFORM = os.getenv("PREFERRED_PLATFORM", "Coinbase").strip() or "Coinbase"

# -------------------- CONFIG --------------------
EXCHANGE_ID = "coinbase"
TIMEFRAME = "15m"
LIMIT = 200

SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "XLM/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "POL/USD",
    "LTC/USD", "DOT/USD", "ATOM/USD", "UNI/USD", "AAVE/USD"
]

EMA_FAST = 20
EMA_SLOW = 50

# RSI filter thresholds (tune to generate more/less signals)
BUY_RSI_MAX = 55
SELL_RSI_MIN = 45

SLEEP_SECONDS = 60
DEDUP_WINDOW_MIN = 30

DB_PATH = BASE_DIR / "data" / "cryptonite.db"

# -------------------- STORAGE --------------------
def ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts);")


def is_duplicate(symbol: str, timeframe: str, side: str, window_min: int) -> bool:
    """Skip if same symbol+tf+side within last window minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    cutoff_iso = cutoff.isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM signals
            WHERE symbol = ? AND timeframe = ? AND side = ? AND ts >= ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (symbol, timeframe, side, cutoff_iso),
        )
        return cur.fetchone() is not None


def insert_signal(
    symbol: str,
    timeframe: str,
    side: str,
    price: float,
    rsi: float,
    ema_fast: float,
    ema_slow: float,
    confidence: float,
    details: str,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO signals (ts, symbol, timeframe, side, price, rsi, ema_fast, ema_slow, confidence, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, symbol, timeframe, side,
                float(price), float(rsi), float(ema_fast), float(ema_slow),
                float(confidence), str(details),
            ),
        )

# -------------------- TELEGRAM --------------------
def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def telegram_allowed_for_chat(chat_id: str) -> bool:
    """
    Future-ready:
    - If PRO_MODE is off -> allow sending normally (or you can block; up to you)
    - If PRO_MODE is on and PRO_IDS is set -> require chat_id in PRO_IDS
    """
    if not PRO_MODE:
        return True
    if not PRO_IDS:
        return True  # PRO_IDS not configured yet; allow for now
    return chat_id in PRO_IDS


def send_telegram(text: str) -> bool:
    if not telegram_enabled():
        return False

    if not telegram_allowed_for_chat(TELEGRAM_CHAT_ID):
        print("[telegram] blocked: chat_id not in PRO_IDS")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=20,
        )
        if r.status_code != 200:
            print(f"[telegram] FAIL status={r.status_code} body={r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[telegram] EXCEPTION: {e}")
        return False


def send_welcome_if_enabled() -> None:
    """Send a one-time 'official' welcome message if SEND_WELCOME_ON_START=1."""
    if not SEND_WELCOME_ON_START:
        return
    if not telegram_enabled():
        print("[telegram] welcome skipped: telegram OFF")
        return

    msg = (
        "✅ Welcome to Cryptonite Pro!\n\n"
        "Your membership is now active. Please keep notifications ON so you don’t miss signals.\n\n"
        f"Preferred trading platform: {PREFERRED_PLATFORM}\n\n"
        "You’ll receive BUY/SELL alerts here as they trigger.\n\n"
        "— Cryptonite ⚡"
    )

    sent = send_telegram(msg)
    print(f"[telegram] welcome message: {'SENT' if sent else 'FAILED'}")
    print("[telegram] Tip: set SEND_WELCOME_ON_START=0 in .env after it sends once.")

# -------------------- SIGNAL LOGIC --------------------
@dataclass
class Signal:
    side: str  # "BUY" or "SELL"
    price: float
    rsi: float
    ema_fast: float
    ema_slow: float
    confidence: float
    details: str


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    df["ema_fast"] = EMAIndicator(close=close, window=EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=close, window=EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    return df


def crossover(prev_fast: float, prev_slow: float, cur_fast: float, cur_slow: float) -> Tuple[bool, bool]:
    bullish = prev_fast <= prev_slow and cur_fast > cur_slow
    bearish = prev_fast >= prev_slow and cur_fast < cur_slow
    return bullish, bearish


def score_confidence(side: str, rsi: float, fast: float, slow: float) -> float:
    base = 50.0
    sep = abs(fast - slow)

    # scale bonus based on price level to avoid insane boosts
    # small/large coins vary; keep it moderate
    sep_bonus = min(25.0, sep * 100.0)

    rsi_bonus = 0.0
    if side == "BUY":
        if rsi <= 45:
            rsi_bonus = 20.0
        elif rsi <= 55:
            rsi_bonus = 10.0
    else:
        if rsi >= 55:
            rsi_bonus = 20.0
        elif rsi >= 45:
            rsi_bonus = 10.0

    conf = base + sep_bonus + rsi_bonus
    return float(max(0.0, min(99.0, conf)))


def detect_signal(df: pd.DataFrame) -> Optional[Signal]:
    if len(df) < 3:
        return None

    df = df.dropna(subset=["ema_fast", "ema_slow", "rsi"]).copy()
    if len(df) < 3:
        return None

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    bull, bear = crossover(prev["ema_fast"], prev["ema_slow"], cur["ema_fast"], cur["ema_slow"])
    rsi = float(cur["rsi"])
    price = float(cur["close"])
    fast = float(cur["ema_fast"])
    slow = float(cur["ema_slow"])

    if bull and rsi <= BUY_RSI_MAX:
        conf = score_confidence("BUY", rsi, fast, slow)
        details = f"EMA{EMA_FAST}>{EMA_SLOW} crossover + RSI({rsi:.1f}) <= {BUY_RSI_MAX}"
        return Signal("BUY", price, rsi, fast, slow, conf, details)

    if bear and rsi >= SELL_RSI_MIN:
        conf = score_confidence("SELL", rsi, fast, slow)
        details = f"EMA{EMA_FAST}<{EMA_SLOW} crossover + RSI({rsi:.1f}) >= {SELL_RSI_MIN}"
        return Signal("SELL", price, rsi, fast, slow, conf, details)

    return None

# -------------------- DATA FETCH --------------------
def make_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    return ex


def fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# -------------------- MAIN LOOP --------------------
def print_banner() -> None:
    print("\n=== Cryptonite Signal Bot ===")
    print(f"Project: {BASE_DIR}")
    print(f"Exchange: {EXCHANGE_ID} | TF: {TIMEFRAME} | Symbols: {len(SYMBOLS)}")
    print(f"DB: {DB_PATH}")
    print(f".env: {ENV_PATH} (exists={ENV_PATH.exists()})")
    print(f"Telegram: {'ON' if telegram_enabled() else 'OFF'}")
    print(f"CHAT loaded: {TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else '(none)'}")
    print(f"PRO_MODE: {'1' if PRO_MODE else '0'}")
    print(f"PRO_IDS: {','.join(sorted(PRO_IDS)) if PRO_IDS else '(none)'}")
    print(f"Dedup window: {DEDUP_WINDOW_MIN} minutes")
    print("-" * 40)


def format_telegram_message(symbol: str, sig: Signal) -> str:
    emoji = "🟢" if sig.side == "BUY" else "🔴"
    return (
        f"{emoji} Cryptonite Signal: {sig.side}\n"
        f"Symbol: {symbol}\n"
        f"TF: {TIMEFRAME}\n"
        f"Price: {sig.price:.6f}\n"
        f"RSI: {sig.rsi:.1f}\n"
        f"EMA{EMA_FAST}/{EMA_SLOW}: {sig.ema_fast:.6f} / {sig.ema_slow:.6f}\n"
        f"Confidence: {sig.confidence:.0f}%\n"
        f"Details: {sig.details}"
    )


def main() -> None:
    ensure_db()
    ex = make_exchange()
    print_banner()

    # One-time welcome message if enabled
    send_welcome_if_enabled()

    while True:
        for symbol in SYMBOLS:
            try:
                df = fetch_ohlcv(ex, symbol, TIMEFRAME, LIMIT)
                df = compute_indicators(df)
                sig = detect_signal(df)

                if not sig:
                    print(f"[{symbol}] No signal. No valid crossover or RSI filter not met")
                    continue

                if is_duplicate(symbol, TIMEFRAME, sig.side, DEDUP_WINDOW_MIN):
                    print(f"[{symbol}] {sig.side} skipped (duplicate within {DEDUP_WINDOW_MIN}m)")
                    continue

                insert_signal(
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                    side=sig.side,
                    price=sig.price,
                    rsi=sig.rsi,
                    ema_fast=sig.ema_fast,
                    ema_slow=sig.ema_slow,
                    confidence=sig.confidence,
                    details=sig.details,
                )

                msg = format_telegram_message(symbol, sig)
                sent = send_telegram(msg)

                print(f"[{symbol}] ✅ SIGNAL {sig.side} | conf={sig.confidence:.0f}% | Telegram={'SENT' if sent else 'OFF/FAIL'}")

            except Exception as e:
                print(f"[{symbol}] Error: {e}")

        print(f"[loop] Sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
