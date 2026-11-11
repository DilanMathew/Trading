"""Strategy lab: vectorised panel features, signals and two portfolio engines
(breakout/swing engine `run`, and periodic momentum `rotation`).

All signals are computed from data up to day t; orders fill at day t+1 open.
"""
import numpy as np
import pandas as pd

COST = 0.0015  # per side: brokerage + STT + slippage (conservative for delivery trades)


def panel(data: dict, bench: str = "^NSEI"):
    """Align all symbols on benchmark dates -> dict of (T, N) arrays."""
    dates = data[bench].index
    syms = [s for s in data if s != bench]
    P = {}
    for k in ("Open", "High", "Low", "Close", "Volume"):
        P[k] = pd.DataFrame({s: data[s][k] for s in syms}).reindex(dates)
    P["syms"], P["dates"] = syms, dates
    P["bench"] = data[bench].reindex(dates)
    return P


def features(P):
    C, H, L, V = P["Close"], P["High"], P["Low"], P["Volume"]
    F = {}
    for n in (10, 20, 50, 150, 200):
        F[f"sma{n}"] = C.rolling(n, min_periods=n).mean()
    F["ema10"] = C.ewm(span=10, adjust=False).mean()
    F["ema21"] = C.ewm(span=21, adjust=False).mean()
    prev = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - prev).abs(), (L - prev).abs()))
    F["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    F["hi52"] = H.rolling(252, min_periods=200).max()
    F["lo52"] = L.rolling(252, min_periods=200).min()
    F["vol50"] = V.rolling(50, min_periods=30).mean()
    F["vol10"] = V.rolling(10, min_periods=8).mean()
    F["turnover"] = (C * V).rolling(50, min_periods=30).mean()
    # IBD-style relative strength: weighted 3/6/9/12m return, ranked 0-100 across the universe each day
    r = lambda n: C / C.shift(n) - 1
    rs_raw = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    F["rs"] = rs_raw.rank(axis=1, pct=True) * 100
    F["mom12_1"] = C.shift(21) / C.shift(252) - 1             # classic 12-1 momentum
    F["mom6"] = r(126)
    F["vola"] = C.pct_change().rolling(126, min_periods=60).std() * np.sqrt(252)
    # Volatility contraction: ranges of three successive windows in the last 60 days
    def rng(a, b):  # % range of window [t-a, t-b)
        hh = H.shift(b).rolling(a - b).max()
        ll = L.shift(b).rolling(a - b).min()
        return (hh - ll) / hh
    F["c1"], F["c2"], F["c3"] = rng(60, 30), rng(30, 12), rng(12, 0)
    F["pivot"] = H.shift(1).rolling(12).max()            # top of the tightest (latest) contraction
    F["base_low"] = L.rolling(12).min()
    # Market health
    b = P["bench"]["Close"]
    F["mkt_ok"] = (b > b.rolling(200).mean()) & (b.rolling(50).mean() > b.rolling(200).mean())
    F["mkt_ok_fast"] = b > b.rolling(50).mean()
    F["breadth"] = (C > F["sma50"]).sum(axis=1) / C.notna().sum(axis=1) * 100
    return F


def trend_template(P, F, rs_min=70):
    """Classic 'stage 2' trend template (8 conditions)."""
    C = P["Close"]
    return (
        (C > F["sma150"]) & (C > F["sma200"]) & (F["sma150"] > F["sma200"])
        & (F["sma200"] > F["sma200"].shift(21)) & (F["sma50"] > F["sma150"]) & (C > F["sma50"])
        & (C >= 1.30 * F["lo52"]) & (C >= 0.75 * F["hi52"]) & (F["rs"] >= rs_min)
    )


def run(P, F, entry, rank, *, slots=8, risk=0.01, max_w=0.15, stop_mode="base", stop_max=0.08,
        exit_mode="ema21", partial_r=None, time_stop=None, mkt="mkt_ok", liquid=2e8,
        start=None, end=None, capital=1_000_000):
    """Generic long-only portfolio engine.

    entry: bool DataFrame (signal on close of t) ; rank: DataFrame, higher = better.
    stop_mode: 'base' (base low, capped at stop_max) | 'atr' (2.5 ATR)
    exit_mode: 'ema21' / 'sma50' close-below trailing exit | 'atr3' chandelier | 'none'
    """
    O, H, L, C = (P[k].values for k in ("Open", "High", "Low", "Close"))
    dates = P["dates"]
    T, N = C.shape
    ent = (entry & (F["turnover"] > liquid)).fillna(False).values
    rk = rank.values
    mk = F[mkt].values if mkt else np.ones(T, bool)
    atr = F["atr"].values
    base_low = F["base_low"].values
    trail = {"ema21": F["ema21"].values, "sma50": F["sma50"].values, "ema10": F["ema10"].values}.get(exit_mode)
    i0 = 260 if start is None else max(260, dates.searchsorted(pd.Timestamp(start)))
    i1 = T if end is None else dates.searchsorted(pd.Timestamp(end))
    cash, pos, eq, trades = float(capital), {}, [], []
    for i in range(i0, i1):
        # --- exits
        for j in list(pos):
            p = pos[j]
            if np.isnan(C[i, j]):
                continue
            px = None
            if L[i, j] <= p["stop"]:
                px = min(O[i, j], p["stop"])
            else:
                if partial_r and not p["half"] and H[i, j] >= p["entry"] + partial_r * p["R"]:
                    q = p["qty"] // 2
                    tp = p["entry"] + partial_r * p["R"]
                    cash += q * tp * (1 - COST); p["proceeds"] += q * tp * (1 - COST); p["qty"] -= q
                    p["half"] = True; p["stop"] = max(p["stop"], p["entry"])
                p["hi"] = max(p["hi"], C[i, j])
                if exit_mode == "atr3":
                    p["stop"] = max(p["stop"], p["hi"] - 3 * atr[i, j])
                elif trail is not None and (i - p["i"]) >= 5 and C[i, j] < trail[i, j]:
                    px = C[i, j]
                if time_stop and (i - p["i"]) >= time_stop and C[i, j] < p["entry"] * 1.02:
                    px = C[i, j]
            if px is not None:
                cash += p["qty"] * px * (1 - COST)
                p["proceeds"] += p["qty"] * px * (1 - COST)
                trades.append((dates[p["i"]], dates[i], p["proceeds"] / p["cost"] - 1, i - p["i"], j))
                del pos[j]
        # --- entries (signal at i-1, fill at open i)
        equity = cash + sum(p["qty"] * (C[i - 1, j] if not np.isnan(C[i - 1, j]) else p["entry"]) for j, p in pos.items())
        if mk[i - 1] and len(pos) < slots:
            cand = np.where(ent[i - 1])[0]
            cand = [j for j in cand if j not in pos and not np.isnan(O[i, j])]
            cand.sort(key=lambda j: -np.nan_to_num(rk[i - 1, j], nan=-1e9))
            for j in cand[: slots - len(pos)]:
                e = O[i, j]
                if stop_mode == "base":
                    st = max(base_low[i - 1, j], e * (1 - stop_max))
                else:
                    st = e - 2.5 * atr[i - 1, j]
                if not (st < e) or np.isnan(st):
                    continue
                if e > C[i - 1, j] * 1.04:  # skip big gap-ups: chasing ruins R:R
                    continue
                qty = int(min(equity * risk / (e - st), equity * max_w / e, cash / (e * (1 + COST))))
                if qty < 1:
                    continue
                cost = qty * e * (1 + COST)
                cash -= cost
                pos[j] = {"i": i, "entry": e, "stop": st, "R": e - st, "qty": qty, "hi": e, "half": False,
                          "cost": cost, "proceeds": 0.0}
        eq.append(cash + sum(p["qty"] * (C[i, j] if not np.isnan(C[i, j]) else p["entry"]) for j, p in pos.items()))
    return stats(pd.Series(eq, index=dates[i0:i1]), trades, P, capital)


def rotation(P, F, rank, eligible, *, top=10, every=21, mkt="mkt_ok", start=None, end=None,
             capital=1_000_000, buffer=2.0, stop_pct=None, liquid=2e8, calendar=None, defensive=None):
    """Periodic rebalance into the top-N ranked eligible stocks, equal weight.
    buffer: keep a holding while its rank is within top*buffer (cuts turnover)."""
    O, C = P["Open"].values, P["Close"].values
    dates = P["dates"]
    T, N = C.shape
    el = (eligible & (F["turnover"] > liquid)).fillna(False).values
    rk = rank.values
    mk = F[mkt].values if mkt else np.ones(T, bool)
    i0 = 260 if start is None else max(260, dates.searchsorted(pd.Timestamp(start)))
    i1 = T if end is None else dates.searchsorted(pd.Timestamp(end))
    cash, pos, eq, trades = float(capital), {}, [], []
    last_rebal = None

    def sell(j, px, i):
        nonlocal cash
        p = pos.pop(j)
        cash += p["qty"] * px * (1 - COST)
        trades.append((dates[p["i"]], dates[i], px * (1 - COST) / (p["entry"] * (1 + COST)) - 1, i - p["i"], j))

    for i in range(i0, i1):
        if stop_pct:
            for j in list(pos):
                if not np.isnan(C[i, j]) and C[i, j] < pos[j]["hi"] * (1 - stop_pct):
                    sell(j, C[i, j], i)
                elif not np.isnan(C[i, j]):
                    pos[j]["hi"] = max(pos[j]["hi"], C[i, j])
        if calendar is not None:
            do = calendar[i]
        else:
            do = (i - i0) % every == 0
        if do:
            s = i - 1
            order = np.argsort(-np.nan_to_num(np.where(el[s], rk[s], np.nan), nan=-1e9))
            order = [j for j in order if el[s, j]]
            keep_set = set(order[: int(top * buffer)])
            target = [] if not mk[s] else order[:top]
            if defensive is not None:
                keep_set.add(defensive)
                if not mk[s]:
                    target = [defensive]
            for j in list(pos):
                if (not mk[s] and j != defensive) or j not in keep_set or (mk[s] and j == defensive):
                    if not np.isnan(O[i, j]):
                        sell(j, O[i, j], i)
            new = [j for j in target if j not in pos][: max(0, top - len(pos))] if target != [defensive] else [j for j in target if j not in pos]
            equity = cash + sum(p["qty"] * np.nan_to_num(C[s, j], nan=p["entry"]) for j, p in pos.items())
            for j in new:
                e = O[i, j]
                if np.isnan(e):
                    continue
                w = equity if j == defensive else equity / top
                qty = int(min(w, cash) / (e * (1 + COST)))
                if qty < 1:
                    continue
                cash -= qty * e * (1 + COST)
                pos[j] = {"i": i, "entry": e, "qty": qty, "hi": e}
        eq.append(cash + sum(p["qty"] * np.nan_to_num(C[i, j], nan=p["entry"]) for j, p in pos.items()))
        if do:
            last_rebal = i
    out = stats(pd.Series(eq, index=dates[i0:i1]), trades, P, capital)
    out["positions"] = {j: dict(p) for j, p in pos.items()}
    out["cash"] = cash
    out["last_rebalance"] = dates[last_rebal] if last_rebal is not None else None
    return out


def stats(curve, trades, P, capital):
    b = P["bench"]["Close"].reindex(curve.index).ffill()
    # Equal-weight buy&hold of the same universe (same survivorship bias -> fairer benchmark)
    C = P["Close"].reindex(curve.index)
    ew = (C / C.bfill().iloc[0]).mean(axis=1)
    yrs = len(curve) / 252
    def cagr(s): return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    def mdd(s): return (s / s.cummax() - 1).min()
    d = curve.pct_change().dropna()
    rets = np.array([t[2] for t in trades]) if trades else np.array([0.0])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    return {
        "cagr": cagr(curve) * 100, "mdd": mdd(curve) * 100,
        "sharpe": d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0,
        "calmar": cagr(curve) / abs(mdd(curve)) if mdd(curve) < 0 else 0,
        "nifty_cagr": cagr(b) * 100, "nifty_mdd": mdd(b) * 100,
        "ew_cagr": cagr(ew) * 100, "ew_mdd": mdd(ew) * 100,
        "trades": len(trades), "win": (rets > 0).mean() * 100 if trades else 0,
        "avg_win": wins.mean() * 100 if len(wins) else 0, "avg_loss": losses.mean() * 100 if len(losses) else 0,
        "pf": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "exposure": None, "curve": curve, "trade_list": trades,
    }


def fmt(name, s):
    return (f"{name:<42} CAGR {s['cagr']:6.1f}%  MDD {s['mdd']:6.1f}%  Sharpe {s['sharpe']:4.2f}  "
            f"Calmar {s['calmar']:4.2f}  trades {s['trades']:4d}  win {s['win']:4.0f}%  PF {s['pf']:4.2f}  "
            f"| Nifty {s['nifty_cagr']:5.1f}% (MDD {s['nifty_mdd']:5.1f}%)  EW {s['ew_cagr']:5.1f}%/{s['ew_mdd']:5.1f}%")


def calendar_mask(dates, freq):
    """True on the first trading day of each week ('W'), fortnight ('2W') or month ('M')."""
    d = pd.Series(dates, index=dates)
    if freq == "M":
        key = d.dt.year * 100 + d.dt.month
    elif freq == "2W":
        iso = d.dt.isocalendar()
        key = iso.year * 100 + iso.week // 2
    else:
        iso = d.dt.isocalendar()
        key = iso.year * 100 + iso.week
    return (key != key.shift(1)).values
