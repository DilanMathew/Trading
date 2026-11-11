"""Download & cache daily OHLCV for the research universe."""
import pickle, sys
from pathlib import Path
import yfinance as yf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.data import _clean
from analyzer.universe import SYMBOLS

out = {}
syms = ["^NSEI"] + SYMBOLS
df = yf.download(syms, period="11y", interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True)
for s in syms:
    try:
        d = _clean(df[s].copy())
        if len(d) > 300:
            out[s] = d
    except Exception as e:
        print("fail", s, e)
path = Path(sys.argv[1] if len(sys.argv) > 1 else "prices.pkl")
pickle.dump(out, open(path, "wb"))
print(len(out), "symbols saved to", path, "missing:", [s for s in syms if s not in out])
