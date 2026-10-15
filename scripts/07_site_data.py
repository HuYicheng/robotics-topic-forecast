"""Robotics forecast, step 7: page data. site/data.json: per topic the two series (2020-2026, standardised shares with
95% intervals), the rule check on robotics text, the 2027 forecast; the backtest predictions per target year; the
backtest, bootstrap and fit summaries. Topics that failed the rule check carry their series but no forecast.
"""
import os, json, importlib.util, contextlib, io, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
s = pd.read_csv(f"{ROOT}/data/series.csv")
ok = [l.strip() for l in open(f"{ROOT}/data/topics_ok.txt")]
prec = pd.read_csv(f"{ROOT}/results/robotics_precision.csv").set_index("topic")
fc = pd.read_csv(f"{ROOT}/results/forecast_2027.csv").set_index("topic")
r2 = lambda v: None if pd.isna(v) else round(float(v), 3)
topics = []
for t, g in s.groupby("topic", sort=False):
    d = {"t": t, "ok": t in ok}
    for f in ("mlcv", "robotics"):
        h = g[g.field == f].sort_values("year")
        d[f] = [[r2(a), r2(b), r2(c)] for a, b, c in zip(h.share, h.lo, h.hi)]
    if t in prec.index:
        p = prec.loc[t]; d["prec"] = [r2(p.precision_2020), r2(p.precision_2024), int(p.pool_2020), int(p.pool_2024)]
    if t in fc.index:
        x = fc.loc[t]
        d["fc"] = {k: r2(x[k]) for k in ("robotics_2027", "lo80", "hi80", "lo95", "hi95", "pred_change_logit", "mlcv_change_logit")}
        d["fc"]["rank"] = int(x["rank"]); d["fc"]["floored"] = bool(x.robotics_2026_floored)
    topics.append(d)
spec = importlib.util.spec_from_file_location("bt", f"{ROOT}/scripts/04_backtest.py"); bt = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bt)
preds = {}
for T in (2024, 2025, 2026):
    p, a = bt.fit_predict(T, 1, "lead"); q, _ = bt.fit_predict(T, 0, "trend2")
    preds[T] = [[t, r2(p[t]), r2(q[t]), r2(a[t])] for t in p.index]
out = {"topics": topics, "years": list(range(2020, 2027)), "bt_preds": preds,
       "backtest": pd.read_csv(f"{ROOT}/results/backtest.csv").to_dict("records"),
       "bootstrap": pd.read_csv(f"{ROOT}/results/bootstrap.csv").to_dict("records"),
       "fit": open(f"{ROOT}/results/forecast_fit.txt").read().strip().split("\n"),
       "n_robotics": s[s.field == "robotics"].groupby("year").n.first().astype(int).tolist()}
json.dump(out, open(f"{ROOT}/site/data.json", "w"), separators=(",", ":"))
print(len(topics), "topics,", sum(t["ok"] for t in topics), "forecast;", os.path.getsize(f"{ROOT}/site/data.json"), "bytes")
html = open(f"{ROOT}/site_src/index.html").read().replace("__DATA__", json.dumps(out, separators=(",", ":")).replace("</", "<\\/"))
open(f"{ROOT}/site/index.html", "w").write(html); print("site/index.html", len(html), "bytes")
