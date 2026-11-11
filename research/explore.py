import pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.lab import *

data = pickle.load(open(sys.argv[1], "rb"))
P = panel(data); F = features(P)
C = P["Close"]
print("dates", P["dates"][0].date(), "->", P["dates"][-1].date(), "symbols", len(P["syms"]))
TT = trend_template(P, F)
IS, OOS = ("2016-12-01", "2021-12-31"), ("2022-01-01", None)

vcp = (TT & (F["c1"] > F["c2"]) & (F["c2"] > F["c3"]) & (F["c3"] < 0.12)
       & (F["vol10"] < F["vol50"]) .shift(1)
       & (C > F["pivot"]) & (P["Volume"] > 1.5 * F["vol50"]))
breakout = TT & (C >= F["hi52"].shift(1)) & (P["Volume"] > 1.5 * F["vol50"])
pullback = TT & (P["Low"] <= F["ema21"] * 1.01) & (C > F["ema21"]) & (C > P["Open"])
print("signal counts: vcp", int(vcp.sum().sum()), "52wk breakout", int(breakout.sum().sum()), "pullback", int(pullback.sum().sum()))

for per_name, (a, b) in (("IN-SAMPLE 2017-2021", IS), ("OUT-OF-SAMPLE 2022-2026", OOS)):
    print("\n==", per_name)
    t = time.time()
    for ex in ("ema21", "sma50", "atr3"):
        print(fmt(f"VCP breakout, exit {ex}", run(P, F, vcp, F["rs"], exit_mode=ex, start=a, end=b)))
    print(fmt("VCP, sma50, partial 2R", run(P, F, vcp, F["rs"], exit_mode="sma50", partial_r=2, start=a, end=b)))
    print(fmt("52wk-high breakout, sma50", run(P, F, breakout, F["rs"], exit_mode="sma50", start=a, end=b)))
    print(fmt("TT pullback to 21EMA, sma50", run(P, F, pullback, F["rs"], exit_mode="sma50", start=a, end=b)))
    for top in (10, 15):
        print(fmt(f"Rotation RS top{top} monthly, TT", rotation(P, F, F["rs"], TT, top=top, start=a, end=b)))
        print(fmt(f"Rotation 12-1 mom top{top}, >sma200", rotation(P, F, F["mom12_1"], C > F["sma200"], top=top, start=a, end=b)))
        print(fmt(f"Rotation mom/vol top{top}, >sma200", rotation(P, F, F["mom6"] / F["vola"], C > F["sma200"], top=top, start=a, end=b)))
    print(fmt("Rotation RS top10, no mkt filter", rotation(P, F, F["rs"], TT, top=10, mkt=None, start=a, end=b)))
    print(f"   ({time.time()-t:.0f}s)")
