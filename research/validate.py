import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.lab import *
import warnings; warnings.filterwarnings("ignore")
SP = sys.argv[1]
data = pickle.load(open(f"{SP}/prices.pkl", "rb")); etf = pickle.load(open(f"{SP}/etf.pkl", "rb"))
data["GOLDBEES.NS"] = etf["GOLDBEES.NS"]
P = panel(data); F = features(P); C = P["Close"]
g = P["syms"].index("GOLDBEES.NS")
F["mkt_either"] = F["mkt_ok"] | (F["breadth"] > 50)
el200 = C > F["sma200"]; el200.iloc[:, g] = False
tt = trend_template(P, F, 0); tt.iloc[:, g] = False
cal = {k: calendar_mask(P["dates"], k) for k in ("W", "2W", "M")}
IS, OOS, ALL = ("2016-12-01", "2021-12-31"), ("2022-01-01", None), ("2016-12-01", None)
cfgs = [
  ("rs >200 top12 2W either buf1.5", dict(rank=F["rs"], eligible=el200, top=12, calendar=cal["2W"], mkt="mkt_either", buffer=1.5)),
  ("rs >200 top10 W either buf1.5", dict(rank=F["rs"], eligible=el200, top=10, calendar=cal["W"], mkt="mkt_either", buffer=1.5)),
  ("rs >200 top15 W either buf3", dict(rank=F["rs"], eligible=el200, top=15, calendar=cal["W"], mkt="mkt_either", buffer=3)),
  ("rs >200 top12 M either buf1.5", dict(rank=F["rs"], eligible=el200, top=12, calendar=cal["M"], mkt="mkt_either", buffer=1.5)),
  ("rs TT top12 2W either buf1.5", dict(rank=F["rs"], eligible=tt, top=12, calendar=cal["2W"], mkt="mkt_either", buffer=1.5)),
  ("rs >200 top12 2W mkt_ok buf1.5", dict(rank=F["rs"], eligible=el200, top=12, calendar=cal["2W"], mkt="mkt_ok", buffer=1.5)),
  ("rs >200 top12 2W NOFILTER", dict(rank=F["rs"], eligible=el200, top=12, calendar=cal["2W"], mkt=None, buffer=1.5)),
  ("rs >200 top12 2W either +GOLD", dict(rank=F["rs"], eligible=el200, top=12, calendar=cal["2W"], mkt="mkt_either", buffer=1.5, defensive=g)),
]
for per, (a, b) in (("IS 2017-21", IS), ("OOS 2022-26", OOS), ("FULL", ALL)):
    print("==", per)
    for n, kw in cfgs:
        kw = dict(kw); r = kw.pop("rank"); e = kw.pop("eligible")
        print(fmt(n, rotation(P, F, r, e, start=a, end=b, **kw)))
# year by year for the main candidate
kw = dict(cfgs[0][1]); r = kw.pop("rank"); e = kw.pop("eligible")
s = rotation(P, F, r, e, start=ALL[0], **kw)
cv = s["curve"]; b = P["bench"]["Close"].reindex(cv.index)
yr = pd.DataFrame({"system": cv.resample("YE").last().pct_change(), "nifty": b.resample("YE").last().pct_change()})
yr.iloc[0] = [cv.resample("YE").last().iloc[0] / cv.iloc[0] - 1, b.resample("YE").last().iloc[0] / b.iloc[0] - 1]
print((yr * 100).round(1).to_string())
print("exposure %:", round((F["mkt_either"].reindex(cv.index)).mean() * 100, 1))
