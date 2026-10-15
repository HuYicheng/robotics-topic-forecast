"""Robotics forecast, step 6: how sure is the lead-1 advantage? Bootstrap over topics (2000 resamples, seed 0) of
Spearman(lead1) - Spearman(baseline) for each target year, baselines trend2 and momentum; predictions as in step 4.
Output: results/bootstrap.csv
"""
import os, importlib.util, contextlib, io, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("bt", f"{ROOT}/scripts/04_backtest.py"); bt = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bt)
rng = np.random.default_rng(0); rows = []
for T in (2024, 2025, 2026):
    p1, act = bt.fit_predict(T, 1, "lead")
    for base in ("trend2", "momentum"):
        pb, _ = bt.fit_predict(T, 0, base)
        P1, PB, A = p1.values, pb.loc[p1.index].values, act.loc[p1.index].values
        obs = spearmanr(P1, A).statistic - spearmanr(PB, A).statistic
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(A), len(A))
            d.append(spearmanr(P1[i], A[i]).statistic - spearmanr(PB[i], A[i]).statistic)
        lo, hi = np.nanquantile(d, [0.025, 0.975])
        rows.append([T, base, round(obs, 3), round(lo, 3), round(hi, 3), round(float(np.mean(np.array(d) <= 0)), 4)])
out = pd.DataFrame(rows, columns=["target_year", "baseline", "rho_lead1_minus_baseline", "lo95", "hi95", "share_le_0"])
out.to_csv(f"{ROOT}/results/bootstrap.csv", index=False); print(out.to_string(index=False))
