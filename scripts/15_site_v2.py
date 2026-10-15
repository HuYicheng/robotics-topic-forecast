"""Forecast v2, step 15: page data (site/data.json) and the page (site/index.html from site_src/index_v2.html)."""
import os, json, importlib.util, contextlib, io, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
def load(p, name):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    with contextlib.redirect_stdout(io.StringIO()): s.loader.exec_module(m)
    return m
rt = load(os.path.expanduser("~/robotics-trends/scripts/04_topics.py"), "rt")
FAM = {n: f.split(" ", 1)[1] for n, f, _ in rt.TOPICS}
v2 = load(f"{ROOT}/scripts/12_backtest_v2.py", "v2")
bt1 = load(f"{ROOT}/scripts/04_backtest.py", "bt1")
r3 = lambda v: None if v is None or pd.isna(v) else round(float(v), 3)
RS = pd.read_csv(f"{ROOT}/data/v2_rob_series.csv"); ES = pd.read_csv(f"{ROOT}/data/v2_emb_series.csv")
MS = pd.read_csv(f"{ROOT}/data/v2_mlcv_split.csv"); FC = pd.read_csv(f"{ROOT}/results/v2_forecast_2027.csv").set_index("key")
EP = pd.read_csv(f"{ROOT}/results/emb_precision.csv").set_index("topic")
RP = pd.read_csv(f"{ROOT}/results/robotics_precision.csv").set_index("topic")
S27 = v2.S_T[2027]
topics = []
for k in v2.KEYS:
    name = k[3:]; f = FC.loc[k]
    d = {"k": k, "t": name, "set": k[:1], "fam": FAM.get(name, "ML/CV topic") if k[0] == "R" else "ML/CV topic",
         "rob": [[r3(a), r3(b), r3(c)] for a, b, c in RS[RS.key == k].sort_values("year")[["share", "lo", "hi"]].values],
         "emb": [r3(v) for v in ES[ES.key == k].sort_values("year").share] if v2.EMB_OK[k] else None,
         "fc": {c: r3(f[c]) for c in ("share_2026", "share_2027", "lo80", "hi80", "lo95", "hi95", "pred_change")},
         "rank": int(f["rank"]), "band": f.size_band, "z": {s: r3(f["z_" + s]) for s in v2.SIG}}
    if k in v2.MLKEY:
        part = S27.at[k, "mlcv_part"]; mk = v2.MLKEY[k]
        d["mlcv"] = [r3(v) for v in MS[(MS.key == mk) & (MS.part == "all")].sort_values("year").share]
        d["mlcv_part"] = part; d["mlcv_from"] = mk[3:]
        if part != "all": d["mlcv_p"] = [r3(v) for v in MS[(MS.key == mk) & (MS.part == part)].sort_values("year").share]
    if k[0] == "R" and name in EP.index: d["eprec"] = [r3(EP.at[name, "prec_2020"]), r3(EP.at[name, "prec_2024"])]
    if k[0] == "M" and name in RP.index: d["rprec"] = [r3(RP.at[name, "precision_2020"]), r3(RP.at[name, "precision_2024"])]
    topics.append(d)
old = []
Mk = [k for k in v2.KEYS if k.startswith("M: ")]
for T in (2024, 2025, 2026):
    p, a = bt1.fit_predict(T, 1, "lead"); tops = [k[3:] for k in Mk]
    c = v2.calfit([t for t in v2.TARGETS if t < T]); ps = c[0] + c[1] * v2.score(v2.S_T[T])[0]
    old.append([T, r3(spearmanr(p[tops], a[tops]).statistic), r3(spearmanr(ps[Mk], v2.dR[T][Mk]).statistic)])
rd = lambda f: pd.read_csv(f"{ROOT}/results/{f}").to_dict("records")
out = {"topics": topics, "years": list(range(2020, 2027)), "backtest": rd("v2_backtest.csv"), "single": rd("v2_single.csv"),
       "ablation": rd("v2_ablation.csv"), "bootstrap": rd("v2_bootstrap.csv"), "emerging_bt": rd("v2_emerging_backtest.csv"),
       "emerging": rd("v2_emerging_2026.csv"), "old_vs_new": old, "fit": open(f"{ROOT}/results/v2_fit.txt").read().strip().split("\n")[:4],
       "n_rob": RS[RS.key == v2.KEYS[0]].sort_values("year").n.astype(int).tolist(),
       "n_emb": ES[ES.key == ES.key.iloc[0]].sort_values("year").n.astype(int).tolist()}
s = json.dumps(out, separators=(",", ":"))
open(f"{ROOT}/site/data.json", "w").write(s)
html = open(f"{ROOT}/site_src/index_v2.html").read().replace("__DATA__", s.replace("</", "<\\/"))
open(f"{ROOT}/site/index.html", "w").write(html); print(len(topics), "topics;", len(s), "bytes data;", len(html), "bytes page")
