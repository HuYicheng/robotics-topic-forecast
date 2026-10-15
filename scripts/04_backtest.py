"""Robotics forecast, step 4: does ML / CV's past change predict robotics' next-year change? (design fixed before
looking at the results, 2026-10-15). Scale: logit of the standardised share (floor 0.05%). Target: robotics change from
year T-1 to T, T in {2024, 2025} (full years); 2026 reported separately (robotics 2026 lacks IROS and CoRL).
Models, each fitted only on target years < T, pooled over topics:
  momentum : robotics change T-2 -> T-1
  trend2   : mean robotics change over T-3 -> T-1
  lead L   : a + b * ML/CV change (T-1-L -> T-L), L = 0, 1, 2   (L = 0: the same-year ML/CV change, known at forecast time? no:
             lead L uses ML/CV changes that ended L years before T, so L >= 1 is available a year ahead; L = 0 shown for reference)
  combined : a + c * robotics change T-2 -> T-1 + b * ML/CV change (T-1-L -> T-L)
Scores per T: Spearman rank correlation across topics, mean absolute error, overlap of the predicted and the actual top 10.
Topics: data/topics_ok.txt if present (passed the robotics-text check), else all with a share >= 0.5% somewhere.
Output: results/backtest.csv, printed table
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
s = pd.read_csv(f"{ROOT}/data/series.csv")
s["z"] = np.log(np.clip(s.share, 0.05, None) / (100 - np.clip(s.share, 0.05, None)))
Z = s.pivot_table(index="topic", columns=["field", "year"], values="z")
ok = f"{ROOT}/data/topics_ok.txt"
topics = [l.strip() for l in open(ok)] if os.path.exists(ok) else \
    [t for t in Z.index if s[s.topic == t].share.max() >= 0.5]
Z = Z.loc[topics]
dR = lambda y: Z[("robotics", y)] - Z[("robotics", y - 1)]
dM = lambda y: Z[("mlcv", y)] - Z[("mlcv", y - 1)]
def design(T, L, kind):
    """rows for target year T: (features, actual)"""
    feats = {}
    if kind in ("momentum", "combined"): feats["mom"] = dR(T - 1)
    if kind == "trend2": feats["tr2"] = (dR(T - 1) + dR(T - 2)) / 2
    if kind in ("lead", "combined"): feats["lead"] = dM(T - L)
    return pd.DataFrame(feats), dR(T)
def fit_predict(T, L, kind):
    if kind == "momentum":
        X, y = design(T, L, kind); return X["mom"], y
    if kind == "trend2":
        X, y = design(T, L, kind); return X["tr2"], y
    tr = [t for t in range(2022, T) if t - L >= 2021 and t - 2 >= 2020]
    Xs, ys = zip(*[design(t, L, kind) for t in tr]) if tr else ([], [])
    if not tr: return None, None
    X = pd.concat(Xs); y = pd.concat(ys); A = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    Xt, yt = design(T, L, kind)
    return pd.Series(np.column_stack([np.ones(len(Xt)), Xt.values]) @ coef, index=Xt.index), yt
rows = []
for T in (2024, 2025, 2026):
    for kind, L in [("momentum", 0), ("trend2", 0), ("lead", 0), ("lead", 1), ("lead", 2), ("combined", 1), ("combined", 2)]:
        pred, act = fit_predict(T, L, kind)
        if pred is None: continue
        rho = spearmanr(pred, act).statistic; mae = float((pred - act).abs().mean())
        top = len(set(pred.nlargest(10).index) & set(act.nlargest(10).index))
        rows.append([T, kind, L, len(act), round(rho, 3), round(mae, 3), top])
out = pd.DataFrame(rows, columns=["target_year", "model", "lag", "topics", "spearman", "mae_logit", "top10_overlap"])
out.to_csv(f"{ROOT}/results/backtest.csv", index=False); print(out.to_string(index=False))
