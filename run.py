#!/usr/bin/env python3
"""Daily run of the Momentum Rotation system + VCP scanner -> docs/data.json / data.js

    python run.py           # real data (Yahoo Finance)
    python run.py --demo    # synthetic data, offline test only
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer import indicators as ind
from analyzer import lab
from analyzer.strategy import reasons
from analyzer.data import demo_data, fetch_bulk
from analyzer.universe import SYMBOLS

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
BT_START = "2016-12-01"
SPLIT = "2022-01-01"   # settings were chosen using data BEFORE this date only


def r2(x, n=2):
    try:
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, n)
    except (TypeError, ValueError):
        return None


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return r2(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def short_stats(s):
    return {k: s[k] for k in ("cagr", "mdd", "sharpe", "trades", "win", "pf", "nifty_cagr", "nifty_mdd", "ew_cagr", "ew_mdd")}


def next_rebalance(last_date, freq):
    days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=40)
    alld = pd.DatetimeIndex([last_date]).append(days)
    m = lab.calendar_mask(alld, freq)
    for d, flag in zip(alld[1:], m[1:]):
        if flag:
            return d
    return days[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    S = cfg["strategy"]
    bench = cfg["benchmark"]
    held = [h["symbol"] for h in cfg.get("portfolio", []) if h.get("quantity", 0) > 0]
    syms = list(dict.fromkeys(SYMBOLS + cfg.get("extra_symbols", []) + held))

    print(f"Loading {len(syms)} symbols ({'DEMO' if args.demo else 'Yahoo'})...")
    if args.demo:
        data, failed = demo_data([bench] + syms, days=2700, seed=args.seed), []
    else:
        data, failed = fetch_bulk([bench] + syms)
    if bench not in data:
        sys.exit("Benchmark failed to load")

    P = lab.panel(data, bench)
    F = lab.features(P)
    C = P["Close"]
    F["mkt_either"] = F["mkt_ok"] | (F["breadth"] > S["breadth_threshold"])
    liquid = S["min_turnover_cr"] * 1e7
    elig = C > F["sma200"]
    cal = lab.calendar_mask(P["dates"], S["rebalance"])
    rot_kw = dict(top=S["top_n"], calendar=cal, mkt="mkt_either", buffer=S["keep_buffer"], liquid=liquid,
                  capital=cfg["capital"])

    print("Backtesting...")
    full = lab.rotation(P, F, F["rs"], elig, start=BT_START, **rot_kw)
    is_ = lab.rotation(P, F, F["rs"], elig, start=BT_START, end=SPLIT, **rot_kw)
    oos = lab.rotation(P, F, F["rs"], elig, start=SPLIT, **rot_kw)
    TT = lab.trend_template(P, F)
    vcp_sig = (TT & (F["c1"] > F["c2"]) & (F["c2"] > F["c3"]) & (F["c3"] < 0.12)
               & (F["vol10"] < F["vol50"]).shift(1) & (C > F["pivot"]) & (P["Volume"] > 1.5 * F["vol50"]))
    bo_sig = TT & (C >= F["hi52"].shift(1)) & (P["Volume"] > 1.5 * F["vol50"])
    comp = []
    for name, fn in (
        ("Momentum Rotation (this system)", lambda a, b: lab.rotation(P, F, F["rs"], elig, start=a, end=b, **rot_kw)),
        ("VCP base breakout", lambda a, b: lab.run(P, F, vcp_sig, F["rs"], exit_mode="ema21", start=a, end=b, liquid=liquid)),
        ("52-week-high breakout", lambda a, b: lab.run(P, F, bo_sig, F["rs"], exit_mode="sma50", start=a, end=b, liquid=liquid)),
    ):
        comp.append({"name": name, "is": short_stats(fn(BT_START, SPLIT)), "oos": short_stats(fn(SPLIT, None))})

    # ---------- today's state ----------
    last = len(P["dates"]) - 1
    today = P["dates"][last]
    syms_p = P["syms"]
    rs_t = F["rs"].iloc[last]
    el_t = (elig & (F["turnover"] > liquid)).iloc[last].fillna(False)
    mkt_on = bool(F["mkt_either"].iloc[last])
    ranked = [s for s in rs_t.sort_values(ascending=False).index if el_t.get(s, False)]
    target = ranked[: S["top_n"]] if mkt_on else []
    keep_zone = set(ranked[: int(S["top_n"] * S["keep_buffer"])]) if mkt_on else set()
    model_pos = {syms_p[j]: p for j, p in full["positions"].items()}
    kept = [s for s in model_pos if s in keep_zone]
    sells = [s for s in model_pos if s not in keep_zone]
    buys = [s for s in target if s not in model_pos][: max(0, S["top_n"] - len(kept))]
    nxt = next_rebalance(today, S["rebalance"])
    per_stock = cfg["capital"] / S["top_n"]
    rank_pos = {s: i + 1 for i, s in enumerate(ranked)}

    def px(s):
        return float(C[s].iloc[last])

    model = []
    for s, p in model_pos.items():
        model.append({"symbol": s, "since": P["dates"][p["i"]], "entry": p["entry"], "price": px(s),
                      "pnl_pct": (px(s) / p["entry"] - 1) * 100, "rank": rank_pos.get(s), "rs": rs_t.get(s),
                      "action": "KEEP" if s in keep_zone else "SELL"})
    model.sort(key=lambda m: m["rank"] or 999)
    plan_buys = [{"symbol": s, "price": px(s), "rank": rank_pos[s], "rs": rs_t[s],
                  "qty": int(per_stock // px(s)), "invest": int(per_stock // px(s)) * px(s)} for s in buys]
    fresh = [{"symbol": s, "price": px(s), "rank": rank_pos[s], "rs": rs_t[s],
              "qty": int(per_stock // px(s)), "invest": int(per_stock // px(s)) * px(s)} for s in target]

    holdings = []
    for h in cfg.get("portfolio", []):
        s = h["symbol"]
        if h.get("quantity", 0) <= 0 or h.get("buy_price", 0) <= 0 or s not in C:
            continue
        p_now = px(s)
        if not mkt_on:
            act, why = "SELL", "Market health is OFF - the system moves to cash. Sell on the next rebalance day."
        elif s in keep_zone:
            act, why = "HOLD", f"Still among the strongest stocks (rank #{rank_pos[s]}). Keep holding."
        elif s in rank_pos:
            act, why = "SELL", f"Dropped to rank #{rank_pos[s]} - outside the top {int(S['top_n']*S['keep_buffer'])}. Sell on the next rebalance day and switch to a stronger stock."
        else:
            act, why = "SELL", "Below its 200-day average (downtrend) or too illiquid. Sell on the next rebalance day."
        holdings.append({"symbol": s, "buy_price": h["buy_price"], "quantity": h["quantity"], "buy_date": h.get("buy_date"),
                         "price": p_now, "pnl_pct": (p_now / h["buy_price"] - 1) * 100,
                         "pnl": (p_now - h["buy_price"]) * h["quantity"], "value": p_now * h["quantity"],
                         "rank": rank_pos.get(s), "action": act, "why": why})

    # ---------- VCP scanner (what BananaPatterns-style screens look for) ----------
    vcp_rows = []
    tt_t = TT.iloc[last].fillna(False)
    for s in syms_p:
        if not tt_t.get(s, False):
            continue
        c1, c2, c3 = F["c1"][s].iloc[last], F["c2"][s].iloc[last], F["c3"][s].iloc[last]
        if c1 > c2 > c3 and c3 < 0.12:
            piv = float(F["pivot"][s].iloc[last])
            vcp_rows.append({"symbol": s, "price": px(s), "pivot": piv, "stop": float(F["base_low"][s].iloc[last]),
                             "to_pivot": (piv / px(s) - 1) * 100, "contractions": [c1 * 100, c2 * 100, c3 * 100],
                             "vol_dry": bool(F["vol10"][s].iloc[last] < F["vol50"][s].iloc[last]),
                             "breakout": bool(vcp_sig[s].iloc[last]), "rs": rs_t.get(s)})
    vcp_rows.sort(key=lambda r: r["to_pivot"])

    # ---------- per-stock table + detail ----------
    tt_parts = {
        "Price above 150 & 200-day avg": (C > F["sma150"]) & (C > F["sma200"]),
        "150-day avg above 200-day avg": F["sma150"] > F["sma200"],
        "200-day avg rising (1 month)": F["sma200"] > F["sma200"].shift(21),
        "50-day avg above 150-day avg": F["sma50"] > F["sma150"],
        "Price above 50-day avg": C > F["sma50"],
        "At least 30% above 52-week low": C >= 1.3 * F["lo52"],
        "Within 25% of 52-week high": C >= 0.75 * F["hi52"],
        "Relative strength rank ≥ 70": F["rs"] >= 70,
    }
    bench_ind = ind.add_all(data[bench])
    stocks = []
    for s in syms_p:
        if s not in data or np.isnan(C[s].iloc[last]):
            continue
        d = ind.add_all(data[s])
        tail = d.tail(260)
        rs_hist = F["rs"][s].reindex(tail.index)
        checks = {k: bool(v[s].iloc[last]) if not pd.isna(v[s].iloc[last]) else False for k, v in tt_parts.items()}
        lastrow = d.iloc[-1]
        status = ("HOLD" if s in model_pos and s in keep_zone else "SELL" if s in model_pos
                  else "BUY" if s in buys else "TOP" if s in target else "")
        stocks.append({
            "symbol": s, "price": px(s), "change_pct": (lastrow["Close"] / d["Close"].iloc[-2] - 1) * 100,
            "rs": rs_t.get(s), "rank": rank_pos.get(s), "tt": sum(checks.values()), "checks": checks,
            "status": status, "vcp": any(v["symbol"] == s for v in vcp_rows),
            "ret": {"1M": lastrow["RET_1M"], "3M": lastrow["RET_3M"], "6M": lastrow["RET_6M"], "12M": lastrow["RET_12M"]},
            "from_high": lastrow["FROM_HIGH"], "rsi": lastrow["RSI"], "vola": lastrow["VOLAT_ANN"],
            "turnover_cr": F["turnover"][s].iloc[last] / 1e7,
            "reasons": reasons(lastrow, bench_ind.iloc[-1]),
            "chart": {"dates": [str(x.date()) for x in tail.index], "close": [r2(x) for x in tail["Close"]],
                      "sma50": [r2(x) for x in tail["SMA50"]], "sma200": [r2(x) for x in tail["SMA200"]],
                      "rs": [r2(x, 0) for x in rs_hist]},
        })

    # ---------- backtest curves ----------
    cv = full["curve"]
    b = P["bench"]["Close"].reindex(cv.index).ffill()
    Cc = C.reindex(cv.index)
    ew = (Cc / Cc.bfill().iloc[0]).mean(axis=1)
    wk = slice(None, None, 5)
    yearly = []
    for y, grp in cv.groupby(cv.index.year):
        bb = b[b.index.year == y]
        start_v = cv[cv.index < grp.index[0]].iloc[-1] if (cv.index < grp.index[0]).any() else grp.iloc[0]
        start_b = b[b.index < grp.index[0]].iloc[-1] if (b.index < grp.index[0]).any() else bb.iloc[0]
        yearly.append({"year": int(y), "system": (grp.iloc[-1] / start_v - 1) * 100, "nifty": (bb.iloc[-1] / start_b - 1) * 100})
    recent_trades = [{"symbol": syms_p[t[4]], "entry": t[0], "exit": t[1], "ret": t[2] * 100, "days": t[3]}
                     for t in full["trade_list"][-15:]][::-1]

    bi = P["bench"].iloc[-1]
    bclose = P["bench"]["Close"]
    out = {
        "version": 2, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "demo": args.demo,
        "market_name": cfg.get("market_name"), "currency": cfg.get("currency", ""), "capital": cfg["capital"],
        "strategy": S, "as_of": today, "next_rebalance": nxt, "last_rebalance": full["last_rebalance"],
        "market": {
            "symbol": bench, "price": bi["Close"], "change_pct": (bclose.iloc[-1] / bclose.iloc[-2] - 1) * 100,
            "on": mkt_on, "index_uptrend": bool(F["mkt_ok"].iloc[last]), "breadth": F["breadth"].iloc[last],
            "breadth_threshold": S["breadth_threshold"],
            "sma50": bclose.rolling(50).mean().iloc[-1], "sma200": bclose.rolling(200).mean().iloc[-1],
            "ret_1m": (bclose.iloc[-1] / bclose.iloc[-22] - 1) * 100, "ret_12m": (bclose.iloc[-1] / bclose.iloc[-253] - 1) * 100,
            "chart": {"dates": [str(x.date()) for x in P["dates"][-260:]], "close": [r2(x) for x in bclose.iloc[-260:]],
                      "sma200": [r2(x) for x in bclose.rolling(200).mean().iloc[-260:]],
                      "breadth": [r2(x, 0) for x in F["breadth"].iloc[-260:]]},
        },
        "model": model, "plan": {"sells": sells, "buys": plan_buys, "keeps": kept, "fresh": fresh,
                                 "per_stock": per_stock},
        "holdings": holdings, "vcp": vcp_rows, "stocks": stocks,
        "backtest": {
            "full": short_stats(full), "is": short_stats(is_), "oos": short_stats(oos), "split": SPLIT,
            "start": cv.index[0], "end": cv.index[-1], "final": cv.iloc[-1], "capital": cfg["capital"],
            "curve": {"dates": [str(x.date()) for x in cv.index[wk]], "system": [r2(x, 0) for x in cv.values[wk]],
                      "nifty": [r2(x, 0) for x in (b / b.iloc[0] * cfg["capital"]).values[wk]],
                      "ew": [r2(x, 0) for x in (ew / ew.iloc[0] * cfg["capital"]).values[wk]]},
            "yearly": yearly, "compare": comp, "recent_trades": recent_trades,
        },
        "failed": failed,
    }
    out = clean(out)
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(out, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    (DOCS / "data.json").write_text(payload)
    (DOCS / "data.js").write_text("window.MARKET_DATA=" + payload + ";\n")
    print(f"Done. Market {'ON' if mkt_on else 'OFF'} | model {len(model)} | sell {sells} | buy {buys}")
    print(f"Backtest full CAGR {full['cagr']:.1f}% MDD {full['mdd']:.1f}% | OOS {oos['cagr']:.1f}% vs Nifty {oos['nifty_cagr']:.1f}%")
    print(f"VCP setups: {len(vcp_rows)} | payload {len(payload)/1e6:.2f} MB | failed {failed}")


if __name__ == "__main__":
    main()
