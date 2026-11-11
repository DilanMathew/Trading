import pickle, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.lab import *
import warnings; warnings.filterwarnings("ignore")
data = pickle.load(open(sys.argv[1], "rb"))
P = panel(data); F = features(P); C = P["Close"]
F["mkt_breadth"] = (F["breadth"] > 40) & F["mkt_ok_fast"].shift(0)
F["mkt_either"] = F["mkt_ok"] | (F["breadth"] > 50)
IS, OOS = ("2016-12-01", "2021-12-31"), ("2022-01-01", None)
rsv = F["rs"] / 100 / F["vola"]  # RS per unit of risk
ranks = {"rs": F["rs"], "rs/vol": rsv.rank(axis=1, pct=True)}
elig = {"TT70": trend_template(P, F, 70), "TT0": trend_template(P, F, 0), ">sma200": C > F["sma200"]}
rows = []
for (rn, r), (en, e), top, every, mkt, buf, stop in itertools.product(
        ranks.items(), elig.items(), (8, 10, 15), (5, 10, 21), ("mkt_ok", "mkt_ok_fast", "mkt_either", None), (1.5, 2.0, 3.0), (None, 0.20)):
    s = rotation(P, F, r, e, top=top, every=every, mkt=mkt, buffer=buf, stop_pct=stop, start=IS[0], end=IS[1])
    rows.append(dict(rank=rn, elig=en, top=top, every=every, mkt=mkt, buf=buf, stop=stop, cagr=s["cagr"], mdd=s["mdd"], sharpe=s["sharpe"], calmar=s["calmar"]))
df = pd.DataFrame(rows)
df.to_pickle(sys.argv[2])
pd.set_option("display.width", 250)
print(df.sort_values("calmar", ascending=False).head(25).to_string())
for k in ("rank", "elig", "top", "every", "mkt", "buf", "stop"):
    print(df.groupby(k, dropna=False)[["cagr", "mdd", "sharpe", "calmar"]].median().round(2))
