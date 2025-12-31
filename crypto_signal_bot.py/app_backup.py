import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hashlib
import uuid

import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
HEADER_IMAGE = ASSETS_DIR / "cryptonite_header.png"

SIGNALS_DB = BASE_DIR / "data" / "cryptonite.db"
SIM_DB = BASE_DIR / "data" / "cryptonite_sim.db"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
EXCHANGE_ID = "coinbase"
TIMEFRAME = "15m"
LIMIT = 200

SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "XLM/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "POL/USD",
    "LTC/USD", "DOT/USD", "ATOM/USD", "UNI/USD", "AAVE/USD"
]
DEFAULT_SYMBOL = "BTC/USD"

TRIAL_DAYS = 7
TRIAL_START_BALANCE = 5000.0

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Cryptonite", layout="wide")

# --------------------------------------------------
# HEADER (Top hero background)
# --------------------------------------------------
def render_header():
    if HEADER_IMAGE.exists():
        # Use base64 to embed (Streamlit-safe)
        import base64
        b64 = base64.b64encode(HEADER_IMAGE.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <style>
              .cryptonite-hero {{
                height: 260px;
                border-radius: 16px;
                background-image: url("data:image/png;base64,{b64}");
                background-size: cover;
                background-position: center;
                margin-bottom: 22px;
                box-shadow: 0 8px 30px rgba(0,0,0,.35);
              }}
              /* remove top padding a bit */
              .block-container {{ padding-top: 1.2rem; }}
            </style>
            <div class="cryptonite-hero"></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Header image not found: {HEADER_IMAGE}")

render_header()

# --------------------------------------------------
# UTIL: stable keys
# --------------------------------------------------
def safe_key(*parts: str) -> str:
    return "k_" + "_".join([p.replace("/", "_").replace(" ", "_") for p in parts])

# --------------------------------------------------
# EXCHANGE + DATA
# --------------------------------------------------
@st.cache_resource
def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    return ex

@st.cache_data(ttl=30)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
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
            prev = float(df["close"].iloc[-30])  # ~7.5h on 15m
            pct = ((last - prev) / prev) * 100.0
            rows.append({"symbol": sym, "last": last, "change_pct": pct})
        except Exception:
            pass
    movers = pd.DataFrame(rows)
    if movers.empty:
        return movers
    return movers.sort_values("change_pct", ascending=False).reset_index(drop=True)

# --------------------------------------------------
# SIGNALS DB
# --------------------------------------------------
@st.cache_data(ttl=5)
def read_signals(limit: int = 2000) -> pd.DataFrame:
    if not SIGNALS_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(SIGNALS_DB) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?;",
            conn,
            params=(int(limit),),
        )
    if not df.empty and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df

# --------------------------------------------------
# SIM DB (Paper trading)
# --------------------------------------------------
def sim_db_init():
    SIM_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SIM_DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT,
            created_at TEXT NOT NULL,
            trial_start TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            user_id TEXT PRIMARY KEY,
            balance REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_price REAL NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0
        )
        """)
        conn.commit()

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_or_create_user(email: str | None) -> str:
    """
    Local-only user system (for now).
    Later we swap this to real OAuth user id.
    """
    sim_db_init()

    # If user already has an id in session, keep it
    if "user_id" in st.session_state and st.session_state["user_id"]:
        return st.session_state["user_id"]

    # Generate deterministic-ish id for email; otherwise random guest id
    if email:
        uid = hashlib.sha256(email.strip().lower().encode()).hexdigest()[:24]
    else:
        uid = "guest_" + uuid.uuid4().hex[:18]

    with sqlite3.connect(SIM_DB) as conn:
        cur = conn.execute("SELECT user_id FROM users WHERE user_id=?;", (uid,))
        row = cur.fetchone()
        if not row:
            created = now_utc_iso()
            trial_start = created
            conn.execute(
                "INSERT INTO users(user_id, email, created_at, trial_start) VALUES(?,?,?,?);",
                (uid, email or "", created, trial_start),
            )
            conn.execute(
                "INSERT INTO accounts(user_id, balance, created_at) VALUES(?,?,?);",
                (uid, float(TRIAL_START_BALANCE), created),
            )
            conn.commit()

    st.session_state["user_id"] = uid
    st.session_state["user_email"] = email or ""
    return uid

def get_user_row(user_id: str) -> dict:
    with sqlite3.connect(SIM_DB) as conn:
        u = conn.execute("SELECT user_id, email, created_at, trial_start FROM users WHERE user_id=?;", (user_id,)).fetchone()
        a = conn.execute("SELECT balance FROM accounts WHERE user_id=?;", (user_id,)).fetchone()
    if not u:
        return {}
    return {
        "user_id": u[0],
        "email": u[1],
        "created_at": u[2],
        "trial_start": u[3],
        "balance": a[0] if a else 0.0,
    }

def trial_status(user_id: str) -> tuple[int, bool]:
    """
    Returns: (days_remaining, active?)
    """
    row = get_user_row(user_id)
    if not row:
        return (0, False)
    ts = pd.to_datetime(row["trial_start"], utc=True, errors="coerce")
    if pd.isna(ts):
        return (0, False)
    end = ts + pd.Timedelta(days=TRIAL_DAYS)
    now = pd.Timestamp.now(tz="UTC")
    remaining = int((end - now).total_seconds() // 86400) + 1
    active = now < end
    return (max(0, remaining), active)

def load_positions(user_id: str) -> pd.DataFrame:
    with sqlite3.connect(SIM_DB) as conn:
        df = pd.read_sql_query(
            "SELECT symbol, qty, avg_price FROM positions WHERE user_id=? ORDER BY symbol;",
            conn, params=(user_id,)
        )
    return df

def load_trades(user_id: str, limit: int = 200) -> pd.DataFrame:
    with sqlite3.connect(SIM_DB) as conn:
        df = pd.read_sql_query(
            "SELECT ts, symbol, side, qty, price, fee FROM trades WHERE user_id=? ORDER BY id DESC LIMIT ?;",
            conn, params=(user_id, int(limit))
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df

def update_balance(user_id: str, new_balance: float):
    with sqlite3.connect(SIM_DB) as conn:
        conn.execute("UPDATE accounts SET balance=? WHERE user_id=?;", (float(new_balance), user_id))
        conn.commit()

def upsert_position(user_id: str, symbol: str, qty: float, avg_price: float):
    with sqlite3.connect(SIM_DB) as conn:
        conn.execute("""
        INSERT INTO positions(user_id, symbol, qty, avg_price)
        VALUES(?,?,?,?)
        ON CONFLICT(user_id, symbol)
        DO UPDATE SET qty=excluded.qty, avg_price=excluded.avg_price
        """, (user_id, symbol, float(qty), float(avg_price)))
        conn.commit()

def delete_position(user_id: str, symbol: str):
    with sqlite3.connect(SIM_DB) as conn:
        conn.execute("DELETE FROM positions WHERE user_id=? AND symbol=?;", (user_id, symbol))
        conn.commit()

def log_trade(user_id: str, symbol: str, side: str, qty: float, price: float, fee: float = 0.0):
    with sqlite3.connect(SIM_DB) as conn:
        conn.execute(
            "INSERT INTO trades(user_id, ts, symbol, side, qty, price, fee) VALUES(?,?,?,?,?,?,?);",
            (user_id, now_utc_iso(), symbol, side.upper(), float(qty), float(price), float(fee)),
        )
        conn.commit()

def execute_trade(user_id: str, symbol: str, side: str, qty: float, price: float) -> tuple[bool, str]:
    """
    Very simple paper trading rules:
    - BUY: must have enough cash
    - SELL: must have enough position qty
    """
    side = side.upper()
    if qty <= 0:
        return False, "Quantity must be > 0"

    row = get_user_row(user_id)
    bal = float(row.get("balance", 0.0))

    pos = load_positions(user_id)
    cur_pos = pos[pos["symbol"] == symbol]
    cur_qty = float(cur_pos["qty"].iloc[0]) if not cur_pos.empty else 0.0
    cur_avg = float(cur_pos["avg_price"].iloc[0]) if not cur_pos.empty else 0.0

    notional = qty * price

    if side == "BUY":
        if notional > bal:
            return False, f"Not enough balance. Need ${notional:,.2f}, have ${bal:,.2f}."
        # New avg price
        new_qty = cur_qty + qty
        new_avg = ((cur_qty * cur_avg) + (qty * price)) / new_qty if new_qty > 0 else price
        update_balance(user_id, bal - notional)
        upsert_position(user_id, symbol, new_qty, new_avg)
        log_trade(user_id, symbol, "BUY", qty, price)
        return True, f"Bought {qty:g} {symbol} @ {price:,.6f}"

    if side == "SELL":
        if qty > cur_qty:
            return False, f"Not enough position to sell. You have {cur_qty:g}."
        new_qty = cur_qty - qty
        update_balance(user_id, bal + notional)
        if new_qty <= 0:
            delete_position(user_id, symbol)
        else:
            # Keep avg price unchanged for remaining shares (simple model)
            upsert_position(user_id, symbol, new_qty, cur_avg)
        log_trade(user_id, symbol, "SELL", qty, price)
        return True, f"Sold {qty:g} {symbol} @ {price:,.6f}"

    return False, "Side must be BUY or SELL"

# --------------------------------------------------
# CHARTS
# --------------------------------------------------
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

def make_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=symbol
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        showlegend=False,
    )
    return fig

# --------------------------------------------------
# AUTH (UI first, local-only)
# --------------------------------------------------
def is_logged_in() -> bool:
    return bool(st.session_state.get("user_id"))

def logout():
    st.session_state["user_id"] = ""
    st.session_state["user_email"] = ""

def auth_panel():
    st.markdown("## Sign in")
    st.write("For now this is **local UI-only**. Next step we’ll wire real Google/Apple login after deployment.")

    c1, c2 = st.columns([0.6, 0.4], gap="large")

    with c1:
        email = st.text_input("Email", value=st.session_state.get("user_email", ""), placeholder="you@email.com")
        if st.button("Sign in with Email (demo)", type="primary", use_container_width=True):
            uid = get_or_create_user(email.strip() if email else None)
            st.success(f"Signed in (demo). User: {uid}")

    with c2:
        st.markdown("### OAuth (coming next)")
        st.button("Continue with Google", disabled=True, use_container_width=True)
        st.button("Continue with Apple", disabled=True, use_container_width=True)
        st.caption("These need a deployed URL for callbacks (we’ll do it when you go public).")

# --------------------------------------------------
# TOP NAV (no sidebar)
# --------------------------------------------------
nav = st.tabs(["Dashboard", "Signals Log", "Simulation", "Pro", "Account"])

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------
if "symbol" not in st.session_state:
    st.session_state.symbol = DEFAULT_SYMBOL

# Create a guest user automatically for Simulation if not logged in
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""
if "user_email" not in st.session_state:
    st.session_state["user_email"] = ""

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
with nav[0]:
    st.markdown("## Dashboard")
    movers = get_movers(SYMBOLS)

    left, main = st.columns([0.30, 0.70], gap="large")

    with left:
        st.subheader("Coins (movers first)")
        q = st.text_input("Search", value="", placeholder="BTC, ETH, SOL...")

        ordered = movers["symbol"].tolist() if not movers.empty else SYMBOLS
        if q.strip():
            ordered = [s for s in ordered if q.upper() in s.upper()]

        # Render as a scroll list of buttons
        box = st.container(height=520, border=True)
        with box:
            for i, sym in enumerate(ordered):
                pct = None
                if not movers.empty and sym in set(movers["symbol"].tolist()):
                    pct = float(movers[movers["symbol"] == sym]["change_pct"].iloc[0])
                label = f"{sym} ({pct:+.2f}%)" if pct is not None else sym
                if st.button(label, use_container_width=True, key=safe_key("coin", str(i), sym)):
                    st.session_state.symbol = sym

    with main:
        sig_all = read_signals(limit=2500)
        last_signal_time = None
        if not sig_all.empty and "ts" in sig_all.columns:
            last_signal_time = sig_all["ts"].dropna().max()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected", st.session_state.symbol)
        k2.metric("Last Signal", "—" if last_signal_time is None else str(last_signal_time)[:19] + "Z")
        if not sig_all.empty and "ts" in sig_all.columns:
            last_24h = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
            k3.metric("Signals (24h)", int((sig_all["ts"] > last_24h).sum()))
        else:
            k3.metric("Signals (24h)", 0)

        if not sig_all.empty and "confidence" in sig_all.columns:
            k4.metric("Avg Confidence (200)", f"{sig_all['confidence'].head(200).mean():.1f}%")
        else:
            k4.metric("Avg Confidence (200)", "—")

        st.divider()

        # Top movers mini charts (fast list)
        st.markdown("### Top Movers (fast list)")
        if movers.empty:
            st.info("Movers unavailable right now (rate limit/connection).")
        else:
            top = movers.head(6).to_dict("records")
            cols = st.columns(3, gap="large")
            for i, row in enumerate(top):
                sym = row["symbol"]
                pct = float(row["change_pct"])
                try:
                    df = fetch_ohlcv(sym).tail(120)
                except Exception:
                    continue
                with cols[i % 3]:
                    st.plotly_chart(
                        mini_line(df, f"{sym} {pct:+.2f}%"),
                        use_container_width=True,
                        key=safe_key("topmini", str(i), sym),
                    )
                    if st.button(f"Open {sym}", use_container_width=True, key=safe_key("open", str(i), sym)):
                        st.session_state.symbol = sym

        st.divider()

        # Main chart
        st.markdown(f"### {st.session_state.symbol} Chart")
        df = fetch_ohlcv(st.session_state.symbol)
        st.plotly_chart(
            make_candlestick(df, st.session_state.symbol),
            use_container_width=True,
            key=safe_key("main", st.session_state.symbol),
        )

# --------------------------------------------------
# SIGNALS LOG
# --------------------------------------------------
with nav[1]:
    st.markdown("## Signals Log")
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

# --------------------------------------------------
# SIMULATION (PUBLIC, 7-day trial, $5k)
# --------------------------------------------------
with nav[2]:
    st.markdown("## Simulation (Paper Trading)")
    st.caption("This is a public demo. Trades are simulated with a $5,000 starting balance per user (trial-style).")

    # ensure user exists (guest is fine)
    if not is_logged_in():
        uid = get_or_create_user(None)  # guest
    else:
        uid = st.session_state["user_id"]

    user = get_user_row(uid)
    days_left, active = trial_status(uid)

    c1, c2, c3 = st.columns([0.33, 0.33, 0.34])
    c1.metric("User", user["email"] if user["email"] else uid)
    c2.metric("Trial Days Remaining", str(days_left))
    c3.metric("Cash Balance", f"${float(user['balance']):,.2f}")

    st.divider()

    left, right = st.columns([0.40, 0.60], gap="large")

    with left:
        st.subheader("Place a simulated trade")

        sym = st.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(st.session_state.symbol) if st.session_state.symbol in SYMBOLS else 0)
        dfp = fetch_ohlcv(sym)
        price = float(dfp["close"].iloc[-1])

        st.write(f"Current price: **{price:,.6f}**")

        side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        qty = st.number_input("Quantity", min_value=0.0, value=0.01, step=0.01, format="%.4f")

        if st.button("Execute (paper)", type="primary", use_container_width=True):
            ok, msg = execute_trade(uid, sym, side, float(qty), float(price))
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)

        st.divider()

        st.subheader("Positions")
        pos = load_positions(uid)
        if pos.empty:
            st.info("No open positions yet.")
        else:
            # Add mark-to-market
            marks = []
            for _, r in pos.iterrows():
                s = r["symbol"]
                try:
                    p = float(fetch_ohlcv(s)["close"].iloc[-1])
                except Exception:
                    p = float(r["avg_price"])
                qtyp = float(r["qty"])
                avg = float(r["avg_price"])
                pnl = (p - avg) * qtyp
                marks.append({"symbol": s, "qty": qtyp, "avg_price": avg, "mark": p, "pnl": pnl})
            mdf = pd.DataFrame(marks).sort_values("pnl", ascending=False)
            st.dataframe(mdf, use_container_width=True, height=360)

    with right:
        st.subheader("Trade History")
        trades = load_trades(uid, limit=200)
        if trades.empty:
            st.info("No trades yet.")
        else:
            st.dataframe(trades, use_container_width=True, height=520)

# --------------------------------------------------
# PRO (UI-only locked)
# --------------------------------------------------
with nav[3]:
    st.markdown("## 🔒 Pro (UI-only for now)")
    st.write("Simulation is public. Telegram signals will be Pro.")

    if not is_logged_in():
        st.info("Sign in to see your Pro status (demo).")

    # For demo: show trial status based on the same user row
    uid = st.session_state["user_id"] if is_logged_in() else get_or_create_user(None)
    days_left, active = trial_status(uid)

    st.markdown("### 7-day Trial (simulation)")
    st.metric("Days remaining", str(days_left))
    st.metric("Trial balance", f"${TRIAL_START_BALANCE:,.2f}")

    st.divider()

    st.markdown("### Telegram signals (locked)")
    st.info("🔐 Upgrade to unlock Telegram BUY/SELL alerts.")
    st.markdown("**Membership:** $49.99 lifetime (payment wiring later)")

# --------------------------------------------------
# ACCOUNT (Login / Logout)
# --------------------------------------------------
with nav[4]:
    st.markdown("## Account")

    if not is_logged_in():
        auth_panel()
    else:
        st.success(f"Signed in as: {st.session_state.get('user_email','(no email)')}  •  {st.session_state['user_id']}")
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()

    st.divider()
    st.caption("Next: real Google/Apple OAuth after deployment (needs callback URL).")
