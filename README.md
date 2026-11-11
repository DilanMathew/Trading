# 📈 Momentum Stock Analyst (NSE)

A rules-based system that tells a beginner **what to buy, what to sell and when**, and shows
everything in a self-updating HTML dashboard.

**Dashboard:** https://dilanmathew.github.io/Trading/ (updated automatically twice every market day)

> ⚠️ **No method can guarantee zero losses.** This system had losing years in testing, and its worst
> peak-to-trough fall was about −23%. It is an educational tool, not SEBI-registered investment advice.
> Start small or paper-trade first.

## How it works: Momentum Rotation
1. **Universe:** ~160 large, liquid NSE stocks (`analyzer/universe.py`).
2. **Rank** each stock by *relative strength*: weighted 3/6/9/12-month return (40/20/20/20),
   turned into a 0–100 percentile across the universe.
3. **Eligible:** price above its 200-day average and average daily turnover ≥ ₹20 cr.
4. **Hold the top 12, equal money in each.** Rebalance on the first trading day of every second week.
   A holding is kept while it ranks in the top 18 (buffer = fewer, cheaper trades).
5. **Market-health switch:** stay invested while the Nifty is in an uptrend (above its 200-day average, with the 50-day above the 200-day)
   **or** more than 50% of the stocks are above their 50-day average. Otherwise sell everything and hold cash.

All settings live in `config.json → strategy`.

## Why this and not the old score system, or BananaPatterns' VCP?
`research/` contains the full study: 162 stocks, Dec 2016 → Nov 2026, 0.15% cost per trade, signals on
day *t* and fills at day *t+1*'s open. Settings were chosen using **only 2017–2021**. **2022 onwards is
the out-of-sample test** (data the design never saw).

| Strategy | CAGR 2017–21 | CAGR 2022–26 (unseen) | Worst fall 2022–26 |
|---|---|---|---|
| **Momentum rotation (this system)** | **+37%** | **+21%** | −23% |
| VCP base breakout (BananaPatterns-style) | +6% | +4% | −20% |
| 52-week-high breakout | +20% | +4% | −32% |
| Nifty 50 | +16% | +8% | −17% |
| Equal-weight basket of the same stocks | +30% | +21% | −22% |

The previous 5-pillar score / swing-trade system lost money (−14%) over 2024–26 and has been retired.

**Honest caveats**
- The universe is *today's* large caps, so stocks that collapsed and left the list are missing. That
  survivorship bias inflates **every** row above. Compare against the equal-weight basket, which has the same
  bias. Out of sample the system **matched** that basket's return, with a much smaller worst fall in
  2017–21 (−21% vs −38%).
- Taxes (STCG) are not deducted. Momentum has bad stretches (2018, 2026 so far).
- Expect real results to be lower than the backtest.

## Your routine
- **On rebalance day** (shown in box 2, every other Monday): sell the SELL list, buy the BUY list with the
  quantity shown (delivery/CNC orders), and leave KEEP stocks alone.
- **On any other day:** do nothing.
- **When box 1 is OFF on a rebalance day:** sell everything and wait in cash or a liquid fund.
- **After every buy:** add it to `config.json → portfolio` so box 4 can track it.

## Run locally
```bash
pip install -r requirements.txt
python run.py          # writes docs/data.json + docs/data.js; open docs/index.html
```
Research scripts:
```bash
python research/fetch.py prices.pkl
python research/explore.py prices.pkl
```

## Files
```
run.py                 daily pipeline → docs/data.json / data.js
analyzer/lab.py        features, trend template, VCP detection, rotation + breakout backtest engines
analyzer/universe.py   stock universe
analyzer/indicators.py per-stock indicators for the detail view
analyzer/strategy.py   plain-English technical notes
analyzer/data.py       Yahoo Finance loaders (+ offline demo data)
research/              parameter grid, out-of-sample validation, strategy exploration
docs/index.html        dashboard (GitHub Pages)
```
