"""Robotics forecast, step 9 (design fixed 2026-10-15 before results, after the user asked whether ML/CV topics and
robotics topics are aligned): does a topic mean the same thing in both fields?
Measure: for topic X, field F and period P (2020-2023, 2024-2026), the log-lift of every other topic Y among papers with
X: log((n_XY + 0.5) / (n_X + 1)) - log((n_Y + 0.5) / (N + 1)). Y = the other 105 rule topics minus "Robotics / embodied
AI"; Y used only when n_XY >= 5 in at least one field. Alignment(X, P) = Spearman across Y of the ML/CV and robotics
log-lifts (needs >= 10 Y). Reference for "unrelated": the same correlation between ML/CV topic X and robotics topic X'
for every X' != X (median over X').
Test (one-sided, expected positive): across the 52 topics, Spearman between alignment (2024-2026) and the advantage of
lead1 over trend2 in the backtest = mean over T in 2024-2026 of |error trend2| - |error lead1|.
Also: alignment change from 2020-2023 to 2024-2026 (a drop = the topic's company changed in one field).
Output: results/alignment.csv
"""
import os, importlib.util, contextlib, io, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = pd.read_csv(f"{ROOT}/data/robotics_mlcv_matrix.csv"); M = pd.read_csv(os.path.expanduser("~/mlcv-trends/data/topic_matrix_ta.csv"))
cols = [c for c in R.columns if c not in ("uid", "venue", "year") and not c.startswith("Robotics /")]
topics = [l.strip() for l in open(f"{ROOT}/data/topics_ok.txt")]
def lifts(df):
    X = df[cols].values.astype(np.int32); C = X.T @ X; n = np.diag(C).astype(float); N = len(df)
    L = np.log((C + 0.5) / (n[:, None] + 1)) - np.log((n[None, :] + 0.5) / (N + 1))
    return pd.DataFrame(L, index=cols, columns=cols), pd.DataFrame(C, index=cols, columns=cols)
per = {"2020-2023": lambda d: d[d.year <= 2023], "2024-2026": lambda d: d[d.year >= 2024]}
res = {}
for p, f in per.items():
    Lm, Cm = lifts(f(M)); Lr, Cr = lifts(f(R))
    def corr(x, xr):
        ys = [y for y in cols if y not in (x, xr) and (Cm.at[x, y] >= 5 or Cr.at[xr, y] >= 5)]
        return spearmanr(Lm.loc[x, ys], Lr.loc[xr, ys]).statistic if len(ys) >= 10 else np.nan
    res[p] = {x: (corr(x, x), np.nanmedian([corr(x, o) for o in topics if o != x])) for x in topics}
spec = importlib.util.spec_from_file_location("bt", f"{ROOT}/scripts/04_backtest.py"); bt = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bt)
gain = sum((bt.fit_predict(T, 0, "trend2")[0] - bt.dR(T)).abs() - (bt.fit_predict(T, 1, "lead")[0] - bt.dR(T)).abs()
           for T in (2024, 2025, 2026)) / 3
out = pd.DataFrame({"topic": topics,
    "align_2020_23": [res["2020-2023"][x][0] for x in topics], "align_2024_26": [res["2024-2026"][x][0] for x in topics],
    "unrelated_median_2024_26": [res["2024-2026"][x][1] for x in topics], "lead1_gain": gain.loc[topics].values})
out["align_change"] = out.align_2024_26 - out.align_2020_23
out = out.sort_values("align_2024_26").round(3); out.to_csv(f"{ROOT}/results/alignment.csv", index=False)
pd.set_option("display.width", 200); print(out.to_string(index=False))
ok = out.dropna(subset=["align_2024_26"]); r = spearmanr(ok.align_2024_26, ok.lead1_gain)
print(f"\nSpearman(alignment 2024-26, lead1 gain) = {r.statistic:.3f}, one-sided p = {r.pvalue / 2 if r.statistic > 0 else 1 - r.pvalue / 2:.3f}, n = {len(ok)}")
print(f"median alignment 2024-26 {ok.align_2024_26.median():.2f}; median unrelated {ok.unrelated_median_2024_26.median():.2f}")
