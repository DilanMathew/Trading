"""Market data loading: Yahoo Finance (real) or a synthetic demo set (offline)."""
import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")]


def fetch_yahoo(symbols, period="3y"):
    import yfinance as yf
    out, failed = {}, []
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            df = _clean(df)
            if len(df) < 260:
                failed.append(s)
                continue
            out[s] = df
        except Exception as e:  # network/symbol errors shouldn't kill the run
            print(f"  ! {s}: {e}")
            failed.append(s)
    return out, failed


def fetch_names(symbols):
    """Best-effort company names (falls back to the ticker)."""
    names = {}
    try:
        import yfinance as yf
        for s in symbols:
            try:
                info = yf.Ticker(s).get_info()
                names[s] = info.get("shortName") or info.get("longName") or s
            except Exception:
                names[s] = s
    except ImportError:
        pass
    return names


def demo_data(symbols, days=780, seed=7):
    """Random-walk prices with regimes - only for testing the pipeline offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    out = {}
    for k, s in enumerate(symbols):
        drift = rng.normal(0.0004, 0.0006)
        vol = rng.uniform(0.008, 0.022)
        regime = np.sin(np.linspace(0, rng.uniform(2, 6), days) + k) * 0.0012
        r = rng.normal(drift + regime, vol, days)
        close = 100 * rng.uniform(1, 30) * np.exp(np.cumsum(r))
        op = close * (1 + rng.normal(0, vol / 3, days))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2, days)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2, days)))
        v = rng.lognormal(13, 0.4, days) * (1 + 3 * np.abs(r))
        out[s] = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": v}, index=idx)
    return out


def fetch_bulk(symbols, period="11y"):
    """One batched Yahoo download for many symbols (much faster, fewer rate limits)."""
    import yfinance as yf
    df = yf.download(symbols, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                     progress=False, threads=True)
    out, failed = {}, []
    for s in symbols:
        try:
            d = _clean(df[s].copy())
            if len(d) >= 260:
                out[s] = d
            else:
                failed.append(s)
        except Exception:
            failed.append(s)
    return out, failed
