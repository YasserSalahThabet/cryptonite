import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
import base64
import uuid
import hashlib

import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go

# ----------------------------
# CONFIG / PATHS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "cryptonite.db"

ASSETS_DIR = BASE_DIR / "assets"
HEADER_IMG = ASSETS_DIR / "cryptonite_header.png"

EXCHANGE_ID = "coinbase"
TIMEFRAME = "15m"
LIMIT = 200

SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "XLM/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "POL/USD",
    "LTC/USD", "DOT/USD", "ATOM/USD", "UNI/USD", "AAVE/USD"
]
DEFAULT_SYMBOL = "BTC/USD"

# Pro/UI-only settings
TRIAL_DAYS = 7
TRIAL_BALANCE = 5000
LIFETIME_PRICE = 49.99

# Win-rate defaults
DEFAULT_TP_PCT = 0.8   # percent
DEFAULT_SL_PCT = 0.5   # percent
DEFAULT_LOOKAHEAD = 12
DEFAULT_MAX_SIGNALS = 300


# ----------------------------
# PAGE CONFIG (NO SIDEBAR)
# ----------------------------
st.set_page_config(page_title="Cryptonite", layout="wide", initial_sidebar_state="collapsed")


# ----------------------------
# HELPERS
# ----------------------------
def safe_key(prefix: str, symbol: str) -> str:
    return f"{prefix}_{symbol.replace('/', '_').replace(' ', '_')}"

def utc_now_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    try:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            return "—"
        return t.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)

def render_header_background():
    if not HEADER_IMG.exists():
        st.markdown("### ⚡ Cryptonite")
        st.caption("15m • Movers first • Simulation is public • Telegram signals are Pro")
        st.divider()
        return

    b64 = base64.b64encode(HEADER_IMG.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
          .cryptonite-hero {{
            height: 240px;
            border-radius: 18px;
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            margin-bottom: 16px;
            box-shadow: 0 10px 28px rgba(0,0,0,.35);
            position: relative;
            overflow: hidden;
          }}
          .cryptonite-hero::after {{
            content:"";
            position:absolute;
            inset:0;
            background: linear-gradient(180deg, rgba(0,0,0,0.08) 0%, rgba(0,0,0,0.70) 100%);
          }}
          .cryptonite-hero-title {{
            position:absolute;
            left:18px;
            bottom:44px;
            z-index:2;
            color:#fff;
            font-weight:800;
            font-size:28px;
            letter-spacing:0.2px;
          }}
          .cryptonite-hero-sub {{
            position:absolute;
            left:18px;
            bottom:18px;
            z-index:2;
            color: rgba(255,255,255,.85);
            font-size:14px;
          }}
          /* hide the default Streamlit sidebar affordance area a bit */
          section[data-testid="stSidebar"] {{ display:none; }}
        </style>

        <div class="cryptonite-hero">
          <div class="cryptonite-hero-title">Cryptonite</div>
          <div class="cryptonite-hero-sub">15m • Movers first • Simulation is public • Telegram signals are Pro</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

def db_exists() -> bool:
    return DB_PATH.exists()

def list_signal_columns() -> set[str]:
    if not db_exists():
        return set()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("PRAGMA table_info(signals);")
        return {row[1] for row in cur.fetchall()}


# ----------------------------
# EXCHANGE + DATA
# ----------------------------
@st.cache_resource
def get_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    return ex

@st.cache_data(ttl=30)
def fetch_ohlcv(symbol: str, limit: int = LIMIT) -> pd.DataFrame:
    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=int(limit))
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

@st.cache_data(ttl=25)
def get_movers(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym)
            if len(df) < 30:
                continue
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-30])
            pct = ((last - prev) / prev) * 100.0
            rows.append({"symbol": sym, "last": last, "change_pct": pct})
        except Exception:
            pass
    movers = pd.DataFrame(rows)
    if movers.empty:
        return movers
    return movers.sort_values("change_pct", ascending=False).reset_index(drop=True)


# ----------------------------
# DB READERS
# ----------------------------
@st.cache_data(ttl=3)
def read_signals(limit: int = 800) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()

    cols = list_signal_columns()
    if not cols:
        return pd.DataFrame()

    base_cols = ["id", "ts", "symbol", "timeframe", "side", "price", "rsi", "ema_fast", "ema_slow", "confidence"]
    if "details" in cols:
        base_cols.append("details")

    select_cols = ", ".join(base_cols)
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT {select_cols} FROM signals ORDER BY id DESC LIMIT {int(limit)};",
            conn,
        )

    if not df.empty and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df

def signals_for_symbol(symbol: str, limit: int = 600) -> pd.DataFrame:
    df = read_signals(limit=limit)
    if df.empty:
        return df
    return df[df["symbol"] == symbol].copy()


# ----------------------------
# PLOTTING
# ----------------------------
def make_candlestick(df: pd.DataFrame, symbol: str, sig_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=symbol
        )
    )

    if not sig_df.empty:
        sig_df = sig_df.sort_values("ts")
        buys = sig_df[sig_df["side"].astype(str).str.upper() == "BUY"]
        sells = sig_df[sig_df["side"].astype(str).str.upper() == "SELL"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["ts"], y=buys["price"], mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=12),
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["ts"], y=sells["price"], mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=12),
                )
            )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        showlegend=True,
    )
    return fig

def mini_line(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], mode="lines"))
    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=title,
    )
    return fig

def render_grid(rows: list[dict], cols_per_row: int, prefix: str):
    cols = st.columns(cols_per_row)
    for i, row in enumerate(rows):
        sym = row["symbol"]
        pct = row.get("change_pct", None)

        try:
            df = fetch_ohlcv(sym).tail(120)
        except Exception:
            continue

        title = sym if pct is None else f"{sym}  {pct:+.2f}%"
        with cols[i % cols_per_row]:
            st.plotly_chart(
                mini_line(df, title),
                use_container_width=True,
                key=safe_key(f"{prefix}_chart_{i}", sym),
            )
            if st.button(
                f"Open {sym}",
                use_container_width=True,
                key=safe_key(f"{prefix}_btn_{i}", sym),
            ):
                st.session_state.symbol = sym


# ----------------------------
# WIN-RATE ESTIMATE (BACKTEST)
# ----------------------------
@st.cache_data(ttl=120)
def fetch_ohlcv_for_eval(symbol: str, limit: int = 600) -> pd.DataFrame:
    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=int(limit))
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def evaluate_signals(
    signals: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    lookahead_candles: int,
    max_signals: int,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    s = signals.dropna(subset=["ts", "symbol", "side", "price"]).copy()
    s = s.sort_values("ts", ascending=False).head(int(max_signals))
    s["side"] = s["side"].astype(str).str.upper()

    results = []
    symbols = sorted(s["symbol"].unique().tolist())

    ohlcv_map = {}
    for sym in symbols:
        try:
            ohlcv_map[sym] = fetch_ohlcv_for_eval(sym, limit=600)
        except Exception:
            ohlcv_map[sym] = pd.DataFrame()

    for _, row in s.iterrows():
        sym = row["symbol"]
        side = row["side"]
        entry = float(row["price"])
        ts = row["ts"]

        df = ohlcv_map.get(sym, pd.DataFrame())
        if df.empty:
            continue

        idx_list = df.index[df["ts"] >= ts].tolist()
        if not idx_list:
            continue

        start_idx = idx_list[0]
        end_idx = min(start_idx + int(lookahead_candles), len(df) - 1)
        future = df.iloc[start_idx:end_idx + 1].copy()
        if future.empty:
            continue

        if side == "BUY":
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
            hit_tp = future[future["high"] >= tp]
            hit_sl = future[future["low"] <= sl]
        elif side == "SELL":
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            hit_tp = future[future["low"] <= tp]
            hit_sl = future[future["high"] >= sl]
        else:
            continue

        outcome = "NO_HIT"
        exit_price = float(future["close"].iloc[-1])

        first_tp_i = hit_tp.index.min() if not hit_tp.empty else None
        first_sl_i = hit_sl.index.min() if not hit_sl.empty else None

        if first_tp_i is not None and first_sl_i is not None:
            outcome = "WIN" if first_tp_i < first_sl_i else "LOSS"
            exit_price = tp if outcome == "WIN" else sl
        elif first_tp_i is not None:
            outcome = "WIN"
            exit_price = tp
        elif first_sl_i is not None:
            outcome = "LOSS"
            exit_price = sl
        else:
            if side == "BUY":
                outcome = "WIN" if exit_price > entry else "LOSS"
            else:
                outcome = "WIN" if exit_price < entry else "LOSS"

        if side == "BUY":
            ret = (exit_price - entry) / entry
        else:
            ret = (entry - exit_price) / entry

        results.append({
            "ts": ts,
            "symbol": sym,
            "side": side,
            "entry": entry,
            "exit": float(exit_price),
            "return_pct": float(ret * 100.0),
            "outcome": outcome,
        })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values("ts", ascending=False)
    return out


# ----------------------------
# TRIAL (UI-only, session)
# ----------------------------
def ensure_trial_start():
    if "trial_start_utc" not in st.session_state:
        st.session_state.trial_start_utc = utc_now_ts()

def get_trial_days_left() -> int:
    ensure_trial_start()
    start = st.session_state.trial_start_utc
    end = start + pd.Timedelta(days=TRIAL_DAYS)
    now = utc_now_ts()
    remaining_days = int((end - now).total_seconds() // 86400) + 1
    return max(0, remaining_days)

def is_trial_active() -> bool:
    ensure_trial_start()
    start = st.session_state.trial_start_utc
    end = start + pd.Timedelta(days=TRIAL_DAYS)
    return utc_now_ts() < end


# ----------------------------
# UI START
# ----------------------------
render_header_background()

# Top navigation (no sidebar)
tab_dash, tab_log, tab_sim, tab_pro, tab_account = st.tabs(
    ["Dashboard", "Signals Log", "Simulation", "Pro", "Account"]
)

# Selected symbol state
if "symbol" not in st.session_state:
    st.session_state.symbol = DEFAULT_SYMBOL

movers = get_movers(SYMBOLS)

# ----------------------------
# DASHBOARD TAB
# ----------------------------
with tab_dash:
    left, main = st.columns([0.28, 0.72], gap="large")

    with left:
        st.subheader("Coins (movers first)")

        if movers.empty:
            ordered = SYMBOLS
            movers_map = {}
            st.info("Movers unavailable. Showing default order.")
        else:
            ordered = movers["symbol"].tolist() + [s for s in SYMBOLS if s not in set(movers["symbol"].tolist())]
            movers_map = {r["symbol"]: r["change_pct"] for r in movers.to_dict("records")}

        q = st.text_input("Search", value="", placeholder="BTC, ETH, SOL...")
        if q.strip():
            ordered = [s for s in ordered if q.upper() in s.upper()]

        for idx, sym in enumerate(ordered):
            pct = movers_map.get(sym)
            is_sel = (sym == st.session_state.symbol)
            label = f"✅ {sym}" if is_sel else sym
            suffix = "" if pct is None else f"  ({pct:+.2f}%)"

            if st.button(label + suffix, use_container_width=True, key=safe_key(f"side_{idx}", sym)):
                st.session_state.symbol = sym

        st.divider()
        st.caption("Movers are last close vs ~7.5h ago (15m candles).")

    with main:
        st.subheader("Dashboard")

        sig_all = read_signals(limit=2500)
        last_signal_time = None
        if not sig_all.empty and "ts" in sig_all.columns:
            last_signal_time = sig_all["ts"].dropna().max()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected", st.session_state.symbol)
        k2.metric("Last Signal", fmt_ts(last_signal_time))

        if not sig_all.empty and "ts" in sig_all.columns:
            last_24h = utc_now_ts() - pd.Timedelta(hours=24)
            k3.metric("Signals (24h)", int((sig_all["ts"] > last_24h).sum()))
        else:
            k3.metric("Signals (24h)", 0)

        if not sig_all.empty and "confidence" in sig_all.columns:
            k4.metric("Avg Confidence (200)", f"{sig_all['confidence'].head(200).mean():.1f}%")
        else:
            k4.metric("Avg Confidence (200)", "—")

        st.divider()

        st.markdown("### Top Movers")
        if movers.empty:
            st.warning("Top movers not available.")
        else:
            render_grid(movers.head(9).to_dict("records"), cols_per_row=3, prefix="top")

        st.divider()

        st.markdown(f"### {st.session_state.symbol} Chart + Signals")
        df = fetch_ohlcv(st.session_state.symbol)
        sig_sym = signals_for_symbol(st.session_state.symbol, limit=900)

        st.plotly_chart(
            make_candlestick(df, st.session_state.symbol, sig_sym),
            use_container_width=True,
            key=safe_key("main_chart", st.session_state.symbol),
        )

        st.markdown("### Recent Signals (selected coin)")
        if sig_sym.empty:
            st.info("No signals logged yet for this coin.")
        else:
            show_cols = [c for c in ["ts", "symbol", "side", "price", "rsi", "ema_fast", "ema_slow", "confidence", "details"] if c in sig_sym.columns]
            st.dataframe(sig_sym[show_cols].head(80), use_container_width=True, height=340)

        st.divider()

        st.markdown("## Win-rate estimate (backtest)")
        st.caption("Estimation only. Uses TP/SL and lookahead candles after each logged signal.")

        colA, colB, colC, colD = st.columns(4)
        with colA:
            tp = st.number_input("Take-profit %", min_value=0.1, max_value=10.0, value=float(DEFAULT_TP_PCT), step=0.1) / 100.0
        with colB:
            sl = st.number_input("Stop-loss %", min_value=0.1, max_value=10.0, value=float(DEFAULT_SL_PCT), step=0.1) / 100.0
        with colC:
            look = st.number_input("Lookahead candles", min_value=1, max_value=200, value=int(DEFAULT_LOOKAHEAD), step=1)
        with colD:
            nmax = st.number_input("Signals to test", min_value=50, max_value=2000, value=int(DEFAULT_MAX_SIGNALS), step=50)

        eval_df = evaluate_signals(sig_all, tp_pct=float(tp), sl_pct=float(sl), lookahead_candles=int(look), max_signals=int(nmax))
        if eval_df.empty:
            st.info("Not enough evaluated data yet.")
        else:
            win_rate = (eval_df["outcome"] == "WIN").mean() * 100.0
            avg_ret = eval_df["return_pct"].mean()
            st.metric("Estimated win rate", f"{win_rate:.1f}%")
            st.metric("Avg return per signal", f"{avg_ret:.2f}%")

            c1, c2 = st.columns(2)
            with c1:
                by_side = eval_df.groupby("side")["outcome"].apply(lambda x: (x == "WIN").mean() * 100).reset_index(name="win_rate_pct")
                st.markdown("### By side")
                st.dataframe(by_side, use_container_width=True, height=160)
            with c2:
                by_sym = eval_df.groupby("symbol")["outcome"].apply(lambda x: (x == "WIN").mean() * 100).reset_index(name="win_rate_pct")
                by_sym = by_sym.sort_values("win_rate_pct", ascending=False)
                st.markdown("### By symbol (top)")
                st.dataframe(by_sym.head(10), use_container_width=True, height=260)

            st.markdown("### Recent evaluated signals")
            st.dataframe(eval_df.head(50), use_container_width=True, height=360)

        st.caption("⚠️ Educational purposes only. Not financial advice.")


# ----------------------------
# SIGNALS LOG TAB
# ----------------------------
with tab_log:
    st.subheader("Signals Log")

    df = read_signals(limit=2000)
    if df.empty:
        st.info("No signals found yet. Make sure the bot is running and writing to the DB.")
    else:
        c1, c2, c3 = st.columns([0.35, 0.25, 0.40])
        with c1:
            sym_filter = st.selectbox("Symbol", ["(All)"] + sorted(df["symbol"].unique().tolist()))
        with c2:
            side_filter = st.selectbox("Side", ["(All)"] + sorted(df["side"].unique().tolist()))
        with c3:
            min_conf = st.slider("Min confidence", 0, 99, 0, 1)

        f = df.copy()
        if sym_filter != "(All)":
            f = f[f["symbol"] == sym_filter]
        if side_filter != "(All)":
            f = f[f["side"] == side_filter]
        if "confidence" in f.columns:
            f = f[f["confidence"] >= min_conf]

        st.dataframe(f, use_container_width=True, height=720)


# ----------------------------
# SIMULATION TAB (placeholder)
# ----------------------------
with tab_sim:
    st.subheader("Simulation (public)")
    st.info("Next step: paper trading engine + 7-day trial + $5,000 starting balance.")


# ----------------------------
# PRO TAB (UI-only)
# ----------------------------
with tab_pro:
    st.subheader("🔒 Cryptonite Pro (UI-only for now)")

    days_left = get_trial_days_left()
    active = is_trial_active()

    c1, c2, c3 = st.columns(3)
    c1.metric("Trial", "Active ✅" if active else "Ended ⛔️")
    c2.metric("Days remaining", str(days_left))
    c3.metric("Lifetime membership", f"${LIFETIME_PRICE:.2f}")

    st.write(
        "Simulation is **unlocked for everyone**. "
        "Telegram buy/sell alerts will be available for **Pro members**."
    )

    st.divider()
    st.success("✅ Simulation: Unlocked (public)")
    st.warning("🔒 Telegram signals: Locked")
    st.write("Upgrade to unlock Telegram buy/sell signals and keep notifications ON.")
    st.button("Upgrade (Coming Soon)", use_container_width=True, disabled=True)

    st.divider()
    st.markdown("### What you get with Pro")
    st.markdown(
        "- Telegram alerts for buy/sell signals (paid)\n"
        "- Priority access + future strategy upgrades\n"
        "- Lifetime membership (one-time)\n"
    )


# ----------------------------
# ACCOUNT TAB (placeholder)
# ----------------------------
with tab_account:
    st.subheader("Account")
    st.info("Next step: Google / Apple / Email login (UI first, backend later).")
