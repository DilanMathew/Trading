"""Historical simulation of the exact same rules used for today's signals.

Rules (no look-ahead - signal on day t's close, act on day t+1's open):
  ENTRY : signal is BUY / STRONG BUY and market regime is not BEAR.
  STOP  : initial stop under swing low (max 2.5 ATR); exit if the day's low hits it.
  T1    : at +2R sell half and move stop to break-even (a winner can't become a loser).
  TRAIL : stop then trails at highest close - 3 ATR.
  EXIT  : also exit if score falls below 45 (trend broken).
"""
import numpy as np
import pandas as pd

from .strategy import BUY, HOLD_MIN, T1_R, TRAIL_ATR, initial_stop

COST_PCT = 0.1  # brokerage + taxes + slippage per side, %


def _entry_ok(score, close, sma50, regime):
    return score >= BUY and close > sma50 and regime != "BEAR"


def backtest_symbol(d: pd.DataFrame, scores: pd.Series, regime: pd.Series) -> dict:
    """Single-stock backtest, 100% of capital per trade. Returns stats + trade list."""
    reg = regime.reindex(d.index).ffill().fillna("CAUTION").values
    sc = scores.values
    o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
    atr, sma50, swing = d["ATR"].values, d["SMA50"].values, d["SWING_LOW"].values
    dates = d.index
    equity = 1.0
    trades = []
    pos = None
    start = int(np.argmax(~np.isnan(sc))) if np.any(~np.isnan(sc)) else len(d)
    eq_curve = []
    for i in range(max(start, 1), len(d)):
        # --- manage open position on day i
        if pos is not None:
            exit_px = None
            if l[i] <= pos["stop"]:
                exit_px = min(o[i], pos["stop"])
            else:
                if not pos["half"] and h[i] >= pos["t1"]:
                    pos["realised"] += 0.5 * (pos["t1"] / pos["entry"] - 1)
                    pos["half"] = True
                    pos["stop"] = max(pos["stop"], pos["entry"])
                pos["hi"] = max(pos["hi"], c[i])
                if pos["half"]:
                    pos["stop"] = max(pos["stop"], pos["hi"] - TRAIL_ATR * atr[i])
                if sc[i] < HOLD_MIN:
                    exit_px = c[i]
            if exit_px is not None:
                w = 0.5 if pos["half"] else 1.0
                ret = pos["realised"] + w * (exit_px / pos["entry"] - 1) - 2 * COST_PCT / 100
                equity *= 1 + ret
                trades.append({
                    "entry_date": str(dates[pos["i"]].date()), "exit_date": str(dates[i].date()),
                    "entry": round(pos["entry"], 2), "exit": round(float(exit_px), 2),
                    "ret_pct": round(ret * 100, 2), "days": int(i - pos["i"]),
                })
                pos = None
        # --- new entry decided on yesterday's close, filled at today's open
        if pos is None and not np.isnan(sc[i - 1]) and _entry_ok(sc[i - 1], c[i - 1], sma50[i - 1], reg[i - 1]):
            entry = o[i]
            stop = initial_stop(entry, atr[i - 1], swing[i - 1])
            if entry > stop:
                pos = {"i": i, "entry": entry, "stop": stop, "t1": entry + T1_R * (entry - stop),
                       "hi": entry, "half": False, "realised": 0.0}
        mtm = equity
        if pos is not None:
            w = 0.5 if pos["half"] else 1.0
            mtm = equity * (1 + pos["realised"] + w * (c[i] / pos["entry"] - 1))
        eq_curve.append(mtm)

    eq = pd.Series(eq_curve) if eq_curve else pd.Series([1.0])
    dd = float((eq / eq.cummax() - 1).min() * 100)
    rets = [t["ret_pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    bh = float((c[-1] / c[start] - 1) * 100) if start < len(c) else 0.0
    years = max((len(d) - start) / 252, 0.25)
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else (None if not wins else 99.0),
        "total_return": round((equity - 1) * 100, 2),
        "cagr": round(((equity) ** (1 / years) - 1) * 100, 2),
        "buy_hold": round(bh, 2),
        "max_drawdown": round(dd, 2),
        "open_trade": pos is not None,
        "recent_trades": trades[-6:],
    }


def backtest_portfolio(data: dict, scores: dict, regime: pd.Series, bench_close: pd.Series, cfg: dict) -> dict:
    """Whole-system simulation: limited slots, risk-based position sizing,
    best-scored candidates first. Compared against simply holding the index."""
    capital0 = float(cfg["capital"])
    risk_pct = cfg["risk_per_trade_pct"] / 100
    max_pos = cfg["max_position_pct"] / 100
    slots = cfg["max_open_positions"]
    syms = list(data)
    dates = sorted(set().union(*[set(data[s].index) for s in syms]))
    dates = pd.DatetimeIndex(dates)
    dates = dates[dates >= dates[0] + pd.Timedelta(days=300)]  # need 200-day history
    reg = regime.reindex(dates).ffill().fillna("CAUTION")
    cash = capital0
    positions = {}
    curve, trades = [], []
    prev_day = None
    for day in dates:
        # 1) manage open positions
        for s in list(positions):
            d = data[s]
            if day not in d.index:
                continue
            row, p = d.loc[day], positions[s]
            exit_px = None
            if row["Low"] <= p["stop"]:
                exit_px = min(row["Open"], p["stop"])
            else:
                if not p["half"] and row["High"] >= p["t1"]:
                    sell = p["qty"] // 2
                    cash += sell * p["t1"] * (1 - COST_PCT / 100)
                    p["pnl"] += sell * (p["t1"] - p["entry"])
                    p["qty"] -= sell
                    p["half"] = True
                    p["stop"] = max(p["stop"], p["entry"])
                p["hi"] = max(p["hi"], row["Close"])
                if p["half"]:
                    p["stop"] = max(p["stop"], p["hi"] - TRAIL_ATR * row["ATR"])
                if scores[s].get(day, 100) < HOLD_MIN:
                    exit_px = row["Close"]
            if exit_px is not None:
                cash += p["qty"] * exit_px * (1 - COST_PCT / 100)
                p["pnl"] += p["qty"] * (exit_px - p["entry"])
                trades.append(p["pnl"] - p["cost"])
                del positions[s]
        # 2) new entries from yesterday's signals
        if prev_day is not None and reg.get(prev_day) != "BEAR" and len(positions) < slots:
            cands = []
            for s in syms:
                if s in positions or prev_day not in data[s].index or day not in data[s].index:
                    continue
                sc = scores[s].get(prev_day, np.nan)
                r = data[s].loc[prev_day]
                if not np.isnan(sc) and _entry_ok(sc, r["Close"], r["SMA50"], reg[prev_day]):
                    cands.append((sc, s))
            equity_now = cash + sum(p["qty"] * data[s]["Close"].get(prev_day, p["entry"]) for s, p in positions.items())
            for sc, s in sorted(cands, reverse=True)[: slots - len(positions)]:
                r_prev, r_today = data[s].loc[prev_day], data[s].loc[day]
                entry = r_today["Open"]
                stop = initial_stop(entry, r_prev["ATR"], r_prev["SWING_LOW"])
                if entry <= stop:
                    continue
                qty = int(min(equity_now * risk_pct / (entry - stop), equity_now * max_pos / entry,
                              cash / (entry * (1 + COST_PCT / 100))))
                if qty < 1:
                    continue
                cost = qty * entry * COST_PCT / 100 * 2
                cash -= qty * entry * (1 + COST_PCT / 100)
                positions[s] = {"qty": qty, "entry": entry, "stop": stop, "t1": entry + T1_R * (entry - stop),
                                "hi": entry, "half": False, "pnl": 0.0, "cost": cost}
        equity = cash + sum(p["qty"] * data[s]["Close"].get(day, p["entry"]) for s, p in positions.items())
        curve.append(equity)
        prev_day = day

    curve = pd.Series(curve, index=dates)
    bench = bench_close.reindex(dates).ffill()
    bench = bench / bench.iloc[0] * capital0
    years = max(len(dates) / 252, 0.25)
    dd = float((curve / curve.cummax() - 1).min() * 100)
    bdd = float((bench / bench.cummax() - 1).min() * 100)
    daily = curve.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    weekly = curve.iloc[::5]
    wins = [t for t in trades if t > 0]
    return {
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "final_equity": round(float(curve.iloc[-1]), 0),
        "total_return": round((curve.iloc[-1] / capital0 - 1) * 100, 2),
        "cagr": round(((curve.iloc[-1] / capital0) ** (1 / years) - 1) * 100, 2),
        "bench_return": round((bench.iloc[-1] / capital0 - 1) * 100, 2),
        "max_drawdown": round(dd, 2), "bench_drawdown": round(bdd, 2),
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "curve": {"dates": [str(x.date()) for x in weekly.index],
                  "equity": [round(float(x), 0) for x in weekly.values],
                  "bench": [round(float(x), 0) for x in bench.iloc[::5].values]},
    }
