"""Technical indicators, all vectorised over a daily OHLCV DataFrame.

Every function returns a pandas Series aligned to the input index so the same
code is used both for "today's" analysis and for the historical backtest
(no look-ahead: each value only uses data up to that day).
"""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(loss.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14):
    """Average Directional Index: trend *strength* (+DI / -DI give direction)."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper, lower = mid + k * sd, mid - k * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def stochastic(df: pd.DataFrame, n=14, d=3):
    ll = df["Low"].rolling(n, min_periods=n).min()
    hh = df["High"].rolling(n, min_periods=n).max()
    k = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every indicator column the strategy needs."""
    d = df.copy()
    c = d["Close"]
    d["SMA20"], d["SMA50"], d["SMA200"] = sma(c, 20), sma(c, 50), sma(c, 200)
    d["EMA20"] = ema(c, 20)
    d["RSI"] = rsi(c)
    d["MACD"], d["MACD_SIG"], d["MACD_HIST"] = macd(c)
    d["ATR"] = atr(d)
    d["ATR_PCT"] = d["ATR"] / c * 100
    d["ADX"], d["PDI"], d["MDI"] = adx(d)
    d["BB_MID"], d["BB_UP"], d["BB_LO"], d["BB_PCTB"] = bollinger(c)
    d["STOCH_K"], d["STOCH_D"] = stochastic(d)
    d["OBV"] = obv(d)
    d["OBV_SLOPE"] = (d["OBV"] - d["OBV"].shift(20)) / d["Volume"].rolling(20).mean().replace(0, np.nan)
    d["VOL_RATIO"] = d["Volume"] / d["Volume"].rolling(50, min_periods=20).mean().replace(0, np.nan)
    d["RET_1M"] = c.pct_change(21) * 100
    d["RET_3M"] = c.pct_change(63) * 100
    d["RET_6M"] = c.pct_change(126) * 100
    d["RET_12M"] = c.pct_change(252) * 100
    d["HIGH_52W"] = d["High"].rolling(252, min_periods=60).max()
    d["LOW_52W"] = d["Low"].rolling(252, min_periods=60).min()
    d["FROM_HIGH"] = (c / d["HIGH_52W"] - 1) * 100
    d["VOLAT_ANN"] = c.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252) * 100
    d["SMA50_SLOPE"] = (d["SMA50"] / d["SMA50"].shift(10) - 1) * 100
    d["SWING_LOW"] = d["Low"].rolling(20, min_periods=10).min()
    d["EXT_EMA20"] = (c / d["EMA20"] - 1) * 100  # how stretched price is above its 20-day average
    return d
