"""Multi-factor scoring model, market-regime filter, signals and trade plans.

The score (0-100) blends five independent "pillars" so that no single
indicator can trigger a trade on its own:

    Trend         30 pts  - is the stock in a long-term uptrend?
    Momentum      25 pts  - is it moving up now, and faster than the market?
    Entry quality 15 pts  - is now a sensible price, or is it over-stretched?
    Volume        10 pts  - are big buyers (volume) supporting the move?
    Risk          20 pts  - how calm is it, and how close to its 52-week high?

A market-regime filter (based on the benchmark index) then blocks new buys
when the overall market is falling - most losses for beginners come from
buying good stocks in a bad market.
"""
import numpy as np
import pandas as pd

PILLARS = {"trend": 30, "momentum": 25, "entry": 15, "volume": 10, "risk": 20}

STRONG_BUY = 75
BUY = 62
HOLD_MIN = 45

# Risk management constants
STOP_ATR = 2.5          # initial stop distance cap in ATRs
TRAIL_ATR = 3.0         # chandelier trailing stop distance
T1_R, T2_R = 2.0, 3.5   # profit targets as multiples of risk (R)


def _b(cond, pts):
    """Boolean series -> points (NaN-safe)."""
    return np.where(pd.Series(cond).fillna(False).astype(bool), pts, 0.0)


def score_frame(d: pd.DataFrame, bench_ret3=None, bench_ret6=None) -> pd.DataFrame:
    """Return a DataFrame of pillar scores + total for every day of `d`."""
    idx = d.index
    c = d["Close"]
    b3 = bench_ret3.reindex(idx).ffill() if bench_ret3 is not None else 0
    b6 = bench_ret6.reindex(idx).ffill() if bench_ret6 is not None else 0

    trend = (
        _b(c > d["SMA200"], 8)
        + _b(d["SMA50"] > d["SMA200"], 7)
        + _b(c > d["SMA50"], 6)
        + _b(d["SMA50_SLOPE"] > 0, 4)
        + _b((d["ADX"] > 20) & (d["PDI"] > d["MDI"]), 5)
    )
    r = d["RSI"]
    momentum = (
        _b((r >= 50) & (r <= 68), 8)
        + _b(((r >= 40) & (r < 50)) | ((r > 68) & (r <= 75)), 4)
        + _b(d["MACD_HIST"] > 0, 5)
        + _b(d["MACD_HIST"] > d["MACD_HIST"].shift(1), 3)
        + _b((d["RET_6M"] - b6) > 0, 5)
        + _b((d["RET_3M"] - b3) > 0, 4)
    )
    ext = d["EXT_EMA20"]
    entry = (
        _b((ext >= -2) & (ext <= 5), 8)
        + _b((ext > 5) & (ext <= 8), 4)
        + _b(ext < -2, 2)
        + _b((d["BB_PCTB"] >= 0.3) & (d["BB_PCTB"] <= 0.9), 4)
        + _b(d["STOCH_K"] < 80, 3)
    )
    volume = _b(d["OBV_SLOPE"] > 0, 6) + _b((d["VOL_RATIO"] > 1.0) & (c > c.shift(1)), 4)
    v = d["VOLAT_ANN"]
    fh = d["FROM_HIGH"]
    ap = d["ATR_PCT"]
    risk = (
        _b(v < 25, 8) + _b((v >= 25) & (v < 35), 5) + _b((v >= 35) & (v < 50), 2)
        + _b(fh > -10, 7) + _b((fh <= -10) & (fh > -20), 4)
        + _b(ap < 2.5, 5) + _b((ap >= 2.5) & (ap < 4), 3)
    )
    out = pd.DataFrame(
        {"trend": trend, "momentum": momentum, "entry": entry, "volume": volume, "risk": risk},
        index=idx,
    )
    out["total"] = out.sum(axis=1)
    out.loc[d["SMA200"].isna(), "total"] = np.nan  # not enough history yet
    return out


def regime_frame(bench: pd.DataFrame) -> pd.Series:
    """BULL / CAUTION / BEAR for every day, from the benchmark index."""
    c, s50, s200 = bench["Close"], bench["SMA50"], bench["SMA200"]
    reg = pd.Series("CAUTION", index=bench.index)
    reg[(c > s200) & (s50 > s200)] = "BULL"
    reg[(c < s200) & (s50 < s200)] = "BEAR"
    reg[s200.isna()] = "CAUTION"
    return reg


def signal_for(score: float, close: float, sma50: float, sma200: float, regime: str) -> str:
    if score is None or np.isnan(score):
        return "NO DATA"
    if score >= STRONG_BUY and close > sma200:
        sig = "STRONG BUY"
    elif score >= BUY and close > sma50:
        sig = "BUY"
    elif score >= HOLD_MIN:
        sig = "HOLD"
    else:
        sig = "AVOID"
    if regime == "BEAR" and sig in ("STRONG BUY", "BUY"):
        return "WAIT"          # good stock, bad market: keep cash
    if regime == "CAUTION" and sig == "STRONG BUY":
        return "BUY"
    if regime == "CAUTION" and sig == "BUY":
        return "WATCH"
    return sig


def initial_stop(entry: float, atr: float, swing_low: float) -> float:
    """Stop just under the recent swing low, but never further than STOP_ATR*ATR
    and never closer than 1 ATR (to avoid being shaken out by noise)."""
    stop = swing_low - 0.2 * atr
    stop = max(stop, entry - STOP_ATR * atr)
    stop = min(stop, entry - 1.0 * atr)
    return stop


def trade_plan(row: pd.Series, cfg: dict) -> dict:
    entry = float(row["Close"])
    atr = float(row["ATR"])
    stop = initial_stop(entry, atr, float(row["SWING_LOW"]))
    risk_ps = entry - stop
    capital = cfg["capital"]
    risk_amt = capital * cfg["risk_per_trade_pct"] / 100
    qty_risk = int(risk_amt // risk_ps) if risk_ps > 0 else 0
    qty_cap = int((capital * cfg["max_position_pct"] / 100) // entry)
    qty = max(0, min(qty_risk, qty_cap))
    return {
        "entry": round(entry, 2),
        "buy_zone_low": round(max(float(row["EMA20"]), entry - 0.5 * atr), 2),
        "buy_zone_high": round(entry + 0.25 * atr, 2),
        "stop": round(stop, 2),
        "stop_pct": round((stop / entry - 1) * 100, 2),
        "target1": round(entry + T1_R * risk_ps, 2),
        "target2": round(entry + T2_R * risk_ps, 2),
        "t1_pct": round(T1_R * risk_ps / entry * 100, 2),
        "t2_pct": round(T2_R * risk_ps / entry * 100, 2),
        "qty": qty,
        "invest": round(qty * entry, 2),
        "max_loss": round(qty * risk_ps, 2),
        "trail_atr": TRAIL_ATR,
    }


def reasons(row: pd.Series, bench_row: pd.Series | None) -> list:
    """Plain-English explanation. Each item: [good(1)/bad(-1)/neutral(0), text]."""
    out = []
    c = row["Close"]
    add = lambda ok, t: out.append([ok, t])
    if c > row["SMA200"]:
        add(1, "Price is above its 200-day average - long-term uptrend.")
    else:
        add(-1, "Price is below its 200-day average - long-term downtrend. Beginners should avoid.")
    if row["SMA50"] > row["SMA200"]:
        add(1, "50-day average is above 200-day ('golden cross' zone).")
    else:
        add(-1, "50-day average is below 200-day ('death cross' zone).")
    if row["ADX"] > 25:
        add(1 if row["PDI"] > row["MDI"] else -1,
            f"Strong trend (ADX {row['ADX']:.0f}) pointing {'UP' if row['PDI'] > row['MDI'] else 'DOWN'}.")
    elif row["ADX"] < 18:
        add(0, f"Weak/sideways trend (ADX {row['ADX']:.0f}) - moves may be choppy.")
    r = row["RSI"]
    if r > 75:
        add(-1, f"RSI {r:.0f}: overbought - price ran up too fast, wait for a dip.")
    elif r >= 50:
        add(1, f"RSI {r:.0f}: healthy buying momentum.")
    elif r < 30:
        add(0, f"RSI {r:.0f}: oversold - may bounce, but falling knives are risky.")
    else:
        add(-1, f"RSI {r:.0f}: momentum is weak.")
    add(1 if row["MACD_HIST"] > 0 else -1,
        "MACD is positive - short-term momentum turning up." if row["MACD_HIST"] > 0
        else "MACD is negative - short-term momentum is down.")
    if bench_row is not None and not np.isnan(row["RET_6M"]):
        diff = row["RET_6M"] - bench_row["RET_6M"]
        add(1 if diff > 0 else -1,
            f"{'Beat' if diff > 0 else 'Lagged'} the market by {abs(diff):.1f}% over 6 months.")
    ext = row["EXT_EMA20"]
    if ext > 8:
        add(-1, f"Price is {ext:.1f}% above its 20-day average - stretched. Better entry likely on a pullback.")
    elif -2 <= ext <= 5:
        add(1, "Price is close to its 20-day average - good low-risk entry area.")
    if row["OBV_SLOPE"] > 0:
        add(1, "Volume flow (OBV) rising - buyers are accumulating.")
    else:
        add(-1, "Volume flow (OBV) falling - sellers are distributing.")
    fh = row["FROM_HIGH"]
    if fh > -5:
        add(1, f"Within {abs(fh):.1f}% of 52-week high - strength.")
    elif fh < -25:
        add(-1, f"{abs(fh):.0f}% below 52-week high - broken stock.")
    v = row["VOLAT_ANN"]
    if v > 45:
        add(-1, f"Very volatile ({v:.0f}%/yr) - use smaller size.")
    elif v < 25:
        add(1, f"Low volatility ({v:.0f}%/yr) - steadier ride.")
    return out
