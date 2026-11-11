#!/usr/bin/env python3
"""Run the full daily analysis and write the dashboard data.

    python run.py            # real market data from Yahoo Finance
    python run.py --demo     # synthetic data (works offline, for testing)

Then open docs/index.html in your browser.
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
from analyzer import strategy as st
from analyzer.backtest import backtest_portfolio, backtest_symbol
from analyzer.data import demo_data, fetch_names, fetch_yahoo

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"


def r2(x):
    try:
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, 2)
    except (TypeError, ValueError):
        return None


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return r2(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def holding_advice(sym, h, d, sig, score):
    """What to do with a stock you already own."""
    since = d[d.index >= pd.Timestamp(h.get("buy_date", d.index[0]))]
    if since.empty:
        since = d.tail(1)
    last = d.iloc[-1]
    price, buy, qty = float(last["Close"]), float(h["buy_price"]), float(h["quantity"])
    hi = float(since["Close"].max())
    atr = float(last["ATR"])
    first = since.iloc[0]
    atr0 = float(first["ATR"]) if not np.isnan(first["ATR"]) else atr
    swing0 = float(first["SWING_LOW"]) if not np.isnan(first["SWING_LOW"]) else buy - 2 * atr0
    init_stop = st.initial_stop(buy, atr0, swing0)
    risk = buy - init_stop
    t1 = buy + st.T1_R * risk
    trail = hi - st.TRAIL_ATR * atr
    stop = max(init_stop, trail) if hi >= t1 else init_stop
    if hi >= t1:
        stop = max(stop, buy)  # once T1 was reached, protect break-even
    pnl_pct = (price / buy - 1) * 100
    if price <= stop:
        action, why = "SELL NOW", f"Price closed below your stop-loss ({stop:.2f}). Exit to protect capital."
    elif score is not None and score < st.HOLD_MIN:
        action, why = "SELL", "The stock's score fell below 45: the uptrend is broken."
    elif last["RSI"] > 80 and pnl_pct > 0:
        action, why = "BOOK PARTIAL PROFIT", "RSI above 80 - very overbought. Sell half, keep the rest with a trailing stop."
    elif hi >= t1:
        action, why = "BOOK 50% / TRAIL", f"Target 1 ({t1:.2f}) was reached. If not done yet, sell half. Stop-loss is now trailing at {stop:.2f}."
    elif sig in ("STRONG BUY",) and pnl_pct > 0:
        action, why = "HOLD / CAN ADD", "Trend is strong and you are in profit."
    else:
        action, why = "HOLD", "Trend intact. Keep holding with the stop-loss below."
    return {
        "symbol": sym, "buy_price": buy, "quantity": qty, "buy_date": h.get("buy_date"),
        "price": r2(price), "pnl_pct": r2(pnl_pct), "pnl": r2((price - buy) * qty),
        "value": r2(price * qty), "stop": r2(stop), "target1": r2(t1),
        "action": action, "why": why, "signal": sig,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline test)")
    ap.add_argument("--seed", type=int, default=7, help="random seed for --demo")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    bench_sym = cfg["benchmark"]
    symbols = list(dict.fromkeys(cfg["watchlist"]))
    held = [h["symbol"] for h in cfg.get("portfolio", []) if h.get("quantity", 0) > 0]
    all_syms = list(dict.fromkeys(symbols + held))

    print(f"Fetching {len(all_syms)} symbols + benchmark ({'DEMO' if args.demo else 'Yahoo'})...")
    if args.demo:
        raw = demo_data([bench_sym] + all_syms, seed=args.seed)
        failed, names = [], {}
    else:
        raw, failed = fetch_yahoo([bench_sym] + all_syms, cfg.get("history_period", "3y"))
        names = fetch_names(list(raw))
    if bench_sym not in raw:
        sys.exit(f"Could not load benchmark {bench_sym}; aborting.")

    bench = ind.add_all(raw.pop(bench_sym))
    regime = st.regime_frame(bench)
    data, scores, results = {}, {}, []
    for s, df in raw.items():
        d = ind.add_all(df)
        sc = st.score_frame(d, bench["RET_3M"], bench["RET_6M"])
        data[s], scores[s] = d, sc["total"]
        last, lsc = d.iloc[-1], sc.iloc[-1]
        reg_now = regime.reindex(d.index).ffill().iloc[-1]
        sig = st.signal_for(lsc["total"], last["Close"], last["SMA50"], last["SMA200"], reg_now)
        bt = backtest_symbol(d, sc["total"], regime)
        # Confidence blends today's score with how well these rules worked on THIS stock.
        conf = lsc["total"] if not np.isnan(lsc["total"]) else 0
        if bt["trades"] >= 3:
            conf = 0.7 * conf + 0.3 * min(100, (bt["win_rate"] or 0) * 0.6 + min(bt["profit_factor"] or 0, 3) / 3 * 40)
        tail = d.tail(260)
        prev = d["Close"].iloc[-2]
        results.append({
            "symbol": s,
            "name": names.get(s, s),
            "price": r2(last["Close"]),
            "change_pct": r2((last["Close"] / prev - 1) * 100),
            "date": str(d.index[-1].date()),
            "score": r2(lsc["total"]),
            "pillars": {k: r2(lsc[k]) for k in st.PILLARS},
            "confidence": r2(conf),
            "signal": sig,
            "plan": st.trade_plan(last, cfg) if sig in ("STRONG BUY", "BUY", "WATCH", "WAIT") else None,
            "reasons": st.reasons(last, bench.iloc[-1]),
            "ind": {k: r2(last[k]) for k in ("RSI", "MACD_HIST", "ADX", "ATR", "ATR_PCT", "SMA20", "SMA50", "SMA200",
                                              "EMA20", "BB_PCTB", "STOCH_K", "VOL_RATIO", "RET_1M", "RET_3M", "RET_6M",
                                              "RET_12M", "HIGH_52W", "LOW_52W", "FROM_HIGH", "VOLAT_ANN", "EXT_EMA20")},
            "backtest": bt,
            "chart": {
                "dates": [str(x.date()) for x in tail.index],
                "close": [r2(x) for x in tail["Close"]],
                "sma50": [r2(x) for x in tail["SMA50"]],
                "sma200": [r2(x) for x in tail["SMA200"]],
                "rsi": [r2(x) for x in tail["RSI"]],
                "score": [r2(x) for x in sc["total"].tail(260)],
            },
        })

    order = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "WAIT": 3, "HOLD": 4, "AVOID": 5, "NO DATA": 6}
    results.sort(key=lambda r: (order.get(r["signal"], 9), -(r["score"] or 0)))

    # Today's suggested allocation: best buys, limited slots, risk-based sizing.
    picks, used = [], 0.0
    for r in results:
        if r["signal"] in ("STRONG BUY", "BUY") and r["plan"] and r["plan"]["qty"] > 0 and len(picks) < cfg["max_open_positions"]:
            if used + r["plan"]["invest"] <= cfg["capital"]:
                picks.append({"symbol": r["symbol"], "name": r["name"], "signal": r["signal"], "score": r["score"],
                              **{k: r["plan"][k] for k in ("entry", "stop", "target1", "target2", "qty", "invest", "max_loss")}})
                used += r["plan"]["invest"]

    by_sym = {r["symbol"]: r for r in results}
    holdings = []
    for h in cfg.get("portfolio", []):
        if h.get("quantity", 0) > 0 and h.get("buy_price", 0) > 0 and h["symbol"] in data:
            r = by_sym[h["symbol"]]
            holdings.append(holding_advice(h["symbol"], h, data[h["symbol"]], r["signal"], r["score"]))

    print("Running portfolio backtest...")
    pbt = backtest_portfolio(data, {s: v.dropna().to_dict() for s, v in scores.items()}, regime, bench["Close"], cfg)

    b = bench.iloc[-1]
    reg_now = regime.iloc[-1]
    advice = {
        "BULL": "Market is in an uptrend. It is OK to buy the stocks marked BUY / STRONG BUY using the stop-losses shown.",
        "CAUTION": "Market is mixed. Buy only STRONG signals, use smaller amounts, keep more cash.",
        "BEAR": "Market is in a downtrend. Do NOT start new buys. Protect capital: keep cash, tighten stops on what you hold.",
    }[reg_now]
    counts = {k: sum(1 for r in results if r["signal"] == k) for k in order}
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demo": args.demo,
        "market_name": cfg.get("market_name", ""),
        "currency": cfg.get("currency", ""),
        "config": {k: cfg[k] for k in ("capital", "risk_per_trade_pct", "max_position_pct", "max_open_positions")},
        "market": {
            "symbol": bench_sym, "price": r2(b["Close"]),
            "change_pct": r2((b["Close"] / bench["Close"].iloc[-2] - 1) * 100),
            "regime": reg_now, "advice": advice,
            "rsi": r2(b["RSI"]), "sma50": r2(b["SMA50"]), "sma200": r2(b["SMA200"]),
            "ret_1m": r2(b["RET_1M"]), "ret_3m": r2(b["RET_3M"]), "ret_12m": r2(b["RET_12M"]),
            "breadth": r2(sum(1 for s in data if data[s]["Close"].iloc[-1] > data[s]["SMA200"].iloc[-1]) / max(len(data), 1) * 100),
            "chart": {"dates": [str(x.date()) for x in bench.tail(260).index],
                      "close": [r2(x) for x in bench["Close"].tail(260)],
                      "sma50": [r2(x) for x in bench["SMA50"].tail(260)],
                      "sma200": [r2(x) for x in bench["SMA200"].tail(260)]},
        },
        "counts": counts,
        "picks": picks,
        "cash_left": r2(cfg["capital"] - used),
        "holdings": holdings,
        "portfolio_backtest": pbt,
        "stocks": results,
        "failed": failed,
    }
    out = clean_json(out)
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(out, separators=(",", ":"), allow_nan=False)
    (DOCS / "data.json").write_text(payload)
    (DOCS / "data.js").write_text("window.MARKET_DATA=" + payload + ";\n")
    print(f"Done. Regime={reg_now}. Picks: {[p['symbol'] for p in picks]}")
    print(f"System backtest: {pbt['total_return']}% vs index {pbt['bench_return']}%, max DD {pbt['max_drawdown']}%")
    if failed:
        print("Failed symbols:", failed)


if __name__ == "__main__":
    main()
