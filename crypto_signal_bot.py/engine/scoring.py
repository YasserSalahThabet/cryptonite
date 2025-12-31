# engine/scoring.py
from __future__ import annotations

def score_signal(
    rsi: float,
    ema_fast: float,
    ema_slow: float,
    price: float,
    side: str,
) -> tuple[float, str]:
    """
    Returns (confidence_percent, explanation).
    A simple Phase 1.5 rule-based scorer (ML-ready later).
    """

    side = side.upper().strip()
    if price <= 0 or ema_fast <= 0 or ema_slow <= 0:
        return 0.0, "Invalid inputs"

    # Base confidence
    conf = 50.0

    # Trend strength (EMA separation)
    ema_sep_pct = abs(ema_fast - ema_slow) / price * 100.0
    # Stronger separation -> higher confidence (cap contribution)
    conf += min(20.0, ema_sep_pct * 4.0)

    # RSI logic
    if side == "BUY":
        # Better if RSI is lower (oversold-ish)
        if rsi <= 30:
            conf += 20
        elif rsi <= 40:
            conf += 12
        elif rsi <= 50:
            conf += 5
        else:
            conf -= 10
    elif side == "SELL":
        # Better if RSI is higher (overbought-ish)
        if rsi >= 70:
            conf += 20
        elif rsi >= 60:
            conf += 12
        elif rsi >= 50:
            conf += 5
        else:
            conf -= 10
    else:
        return 0.0, "Unknown side"

    # Clamp
    conf = max(0.0, min(100.0, conf))

    explanation = (
        f"EMA sep {ema_sep_pct:.2f}% | RSI {rsi:.1f} | side {side} -> {conf:.0f}%"
    )
    return conf, explanation
