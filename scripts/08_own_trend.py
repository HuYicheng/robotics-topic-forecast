"""Robotics forecast, step 8 (design fixed 2026-10-15 before results, after the user asked whether robotics' own curves
already show the trends): step 4 tested only one- and two-year robotics changes. Here, longer robotics-only trends.
Same scale, topics, targets (2024, 2025, 2026 partial) and metrics as step 4.
Own-trend models (robotics only, nothing from ML/CV):
  slope3   : OLS slope of the robotics logit over years T-3..T-1, used as the forecast change
  slopeall : OLS slope over 2020..T-1
  slopeall_cal : a + b * slopeall, a and b fitted on earlier target years (2023..T-1)
Reference: lead1 (step 4). Combination: a + b * ML/CV change (T-1) + c * slopeall, fitted on earlier target years.
Bootstrap over topics (2000, seed 0) of rho(lead1) - rho(best own-trend model of that year).
Output: results/own_trend.csv, results/own_trend_bootstrap.csv
"""
import os, importlib.util, contextlib, io, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("bt", f"{ROOT}/scripts/04_backtest.py"); bt = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bt)
Z, dR, dM = bt.Z, bt.dR, bt.dM
def slope(T, k):
    ys = list(range(T - k if k else 2020, T)); x = np.array(ys) - np.mean(ys)
    M = np.column_stack([Z[("robotics", y)] for y in ys])
    return pd.Series(M @ x / (x @ x), index=Z.index)
def fitted(T, feats):
    tr = range(2023, T)
    A = pd.concat([pd.DataFrame({k: f(t) for k, f in feats.items()}) for t in tr]); y = pd.concat([dR(t) for t in tr])
    X = np.column_stack([np.ones(len(A)), A.values]); c, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    At = pd.DataFrame({k: f(T) for k, f in feats.items()})
    return pd.Series(np.column_stack([np.ones(len(At)), At.values]) @ c, index=At.index)
rows, preds = [], {}
for T in (2024, 2025, 2026):
    act = dR(T)
    P = {"slope3": slope(T, 3), "slopeall": slope(T, 0),
         "slopeall_cal": fitted(T, {"s": lambda t: slope(t, 0)}),
         "lead1": bt.fit_predict(T, 1, "lead")[0],
         "lead1+slopeall": fitted(T, {"m": lambda t: dM(t - 1), "s": lambda t: slope(t, 0)})}
    preds[T] = P
    for k, p in P.items():
        rows.append([T, k, round(spearmanr(p, act).statistic, 3), round(float((p - act).abs().mean()), 3),
                     len(set(p.nlargest(10).index) & set(act.nlargest(10).index))])
out = pd.DataFrame(rows, columns=["target_year", "model", "spearman", "mae_logit", "top10_overlap"])
out.to_csv(f"{ROOT}/results/own_trend.csv", index=False); print(out.to_string(index=False))
rng = np.random.default_rng(0); bs = []
for T in (2024, 2025, 2026):
    own = out[(out.target_year == T) & out.model.isin(["slope3", "slopeall", "slopeall_cal"])].sort_values("spearman").iloc[-1].model
    A = dR(T).values; L = preds[T]["lead1"].loc[dR(T).index].values; O = preds[T][own].loc[dR(T).index].values
    d = [spearmanr(L[i], A[i]).statistic - spearmanr(O[i], A[i]).statistic for i in (rng.integers(0, len(A), len(A)) for _ in range(2000))]
    bs.append([T, own, round(spearmanr(L, A).statistic - spearmanr(O, A).statistic, 3), *np.round(np.nanquantile(d, [0.025, 0.975]), 3)])
bs = pd.DataFrame(bs, columns=["target_year", "best_own_model", "rho_lead1_minus_own", "lo95", "hi95"])
bs.to_csv(f"{ROOT}/results/own_trend_bootstrap.csv", index=False); print(bs.to_string(index=False))
