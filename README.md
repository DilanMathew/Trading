# 📈 Daily Stock Analyst

An automated stock-analysis system for beginners. Every market day it:

1. **Reads the market mood**: BULL / CAUTION / BEAR, based on the Nifty 50 index.
2. **Scores every stock in your watchlist from 0 to 100** using 20+ indicators grouped into 5 pillars.
3. **Tells you what to buy**, how many shares, where to put the **stop-loss** and where to **take profit**.
4. **Tells you when to sell** the stocks you already own.
5. **Backtests the same rules on ~3 years of history** so you can see whether they would have worked.
6. Shows everything in a live **HTML dashboard** that refreshes on its own.

> ⚠️ **No algorithm can guarantee zero losses.** Anyone promising that is lying. This system aims for
> *small, controlled losses and larger profits*: 1% risk per trade, stop-losses on every position,
> profit-taking with trailing stops, and no new buying in falling markets. It is an educational tool,
> not SEBI-registered investment advice. Start small or paper-trade first.

---

## 🖥️ See the dashboard

**Online (recommended):** after a one-time setup (below), open
`https://<your-github-username>.github.io/Trading/`.

**On your computer:**
```bash
pip install -r requirements.txt
python run.py            # downloads real prices and analyses them
# then double-click docs/index.html to open it in your browser
```
`python run.py --demo` runs on synthetic data and needs no internet. Use it only for testing.

## ⚙️ One-time GitHub setup (2 minutes)

1. Repo **Settings → Pages → Build and deployment → Source: _GitHub Actions_**.
2. Repo **Settings → Actions → General → Workflow permissions → _Read and write permissions_** → Save.
3. **Actions** tab → *Daily stock analysis* → **Run workflow**.

After that it runs on its own every hour during Indian market hours (Mon–Fri, 09:45–16:15 IST),
commits fresh data, and republishes the page. The open page reloads its data every 5 minutes.

## 📅 Your daily routine

| Step | Where | What to do |
|---|---|---|
| 1 | Box 1: Market mood | 🟢 BULL = buying is OK · 🟠 CAUTION = small size, strong signals only · 🔴 BEAR = **no new buys** |
| 2 | Box 3: My holdings | **SELL NOW** → sell today. **BOOK 50%** → sell half and keep the rest. **HOLD** → do nothing. |
| 3 | Box 2: Buy today | Buy the quantity shown near the entry price, then **immediately place a stop-loss / GTT order** at the stop price in your broker app. |
| 4 | `config.json` | Add every buy to `portfolio` so the dashboard can track it. |

## 🧠 How the algorithm works

### Score (0–100) = five independent pillars
| Pillar | Points | What it checks |
|---|---|---|
| Trend | 30 | Price vs 50/200-day averages, golden cross, 50-day slope, ADX direction and strength |
| Momentum | 25 | RSI in the healthy 50–68 zone, MACD and its acceleration, 3- and 6-month performance against the index |
| Entry quality | 15 | Distance from the 20-day EMA (avoids chasing), Bollinger %B, Stochastic |
| Volume | 10 | On-Balance-Volume trend, above-average volume on up days |
| Risk | 20 | Annual volatility, distance from 52-week high, ATR % |

**≥75 STRONG BUY · ≥62 BUY · 45–62 HOLD · <45 AVOID/SELL**

**Market filter:** in a BEAR market, every BUY becomes **WAIT**. In CAUTION, signals are downgraded by one level.

**Confidence** = 70% of today's score + 30% of how well these rules did on that stock in the past (win rate and profit factor).

### Risk management (this is what limits losses)
- **Stop-loss:** just under the 20-day swing low, at most 2.5×ATR and at least 1×ATR below entry.
- **Position size:** `capital × 1% ÷ (entry − stop)`, capped at 20% of capital per stock and 6 open positions.
  If a stop is hit, you lose about 1% of your capital.
- **Target 1 = entry + 2R:** sell half and move the stop to break-even. After that, the trade cannot become a loss.
- **Target 2 = entry + 3.5R.** The rest follows a **trailing stop** at highest close − 3×ATR.
- **Exit early** if the score drops below 45 (the trend is broken).

### Backtests
- **Per stock:** the exact rules above, with signals on day *t* and fills at the open of day *t+1* (no look-ahead) and 0.1% cost per side.
- **Whole system:** all stocks together with your capital, slot limit and risk sizing, ranked by score,
  compared with buy-and-hold of the index (return, CAGR, max drawdown, Sharpe, win rate).

## 🔧 Customise: `config.json`
| Key | Meaning |
|---|---|
| `watchlist` | Yahoo symbols. NSE stocks end in `.NS` (e.g. `RELIANCE.NS`), BSE in `.BO`, US stocks have no suffix (e.g. `AAPL`). |
| `benchmark` | Market index for the regime filter (`^NSEI` Nifty 50, `^GSPC` S&P 500) |
| `capital` | Money you plan to invest |
| `risk_per_trade_pct` | Maximum loss per trade as % of capital (keep 0.5–1%) |
| `max_position_pct`, `max_open_positions` | Diversification limits |
| `portfolio` | Your actual buys: `symbol`, `buy_price`, `quantity`, `buy_date` |

## 📁 Project layout
```
run.py                  entry point: fetch → analyse → backtest → docs/data.json + data.js
analyzer/indicators.py  SMA/EMA, RSI, MACD, ATR, ADX/DI, Bollinger, Stochastic, OBV, returns, volatility
analyzer/strategy.py    5-pillar score, market regime, signals, stop/target/size, plain-English reasons
analyzer/backtest.py    per-stock and whole-portfolio simulations
analyzer/data.py        Yahoo Finance loader + offline demo data
docs/index.html         the dashboard (no build step, works offline)
.github/workflows/      scheduled automation + GitHub Pages deploy
```
