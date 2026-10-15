"""Robotics forecast, step 5: forecast the robotics 2027 change for the topics in data/topics_ok.txt with the lead-1
model chosen in step 4 (a + b * ML/CV change of the previous year, on the logit of the standardised share, floor 0.05%).
Fit: pooled over topics, target years 2022-2026 (2026 partial: robotics lacks IROS and CoRL; also fitted without it).
Interval: empirical quantiles of the out-of-sample residuals of the backtest (target years 2024-2026, each predicted from
earlier years only); coverage checked leave-one-year-out (interval from the other two years, coverage on the third).
Output: results/forecast_2027.csv, results/forecast_fit.txt
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
s = pd.read_csv(f"{ROOT}/data/series.csv")
s["z"] = np.log(np.clip(s.share, 0.05, None) / (100 - np.clip(s.share, 0.05, None)))
Z = s.pivot_table(index="topic", columns=["field", "year"], values="z")
S = s.pivot_table(index="topic", columns=["field", "year"], values="share")
topics = [l.strip() for l in open(f"{ROOT}/data/topics_ok.txt")]
Z = Z.loc[topics]; S = S.loc[topics]
dR = lambda y: Z[("robotics", y)] - Z[("robotics", y - 1)]
dM = lambda y: Z[("mlcv", y)] - Z[("mlcv", y - 1)]
inv = lambda z: 100 / (1 + np.exp(-z))
def fit(years):
    x = pd.concat([dM(t - 1) for t in years]); y = pd.concat([dR(t) for t in years])
    A = np.column_stack([np.ones(len(x)), x.values]); coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    return coef
# out-of-sample residuals (same as the backtest: target T fitted on 2022..T-1)
res = {}
for T in (2024, 2025, 2026):
    a, b = fit(range(2022, T)); res[T] = dR(T) - (a + b * dM(T - 1))
lines = []
for lv in (0.8, 0.95):
    cov = []
    for T in res:
        other = pd.concat([res[u] for u in res if u != T])
        lo, hi = other.quantile((1 - lv) / 2), other.quantile((1 + lv) / 2)
        cov.append(f"{T}: {((res[T] >= lo) & (res[T] <= hi)).mean():.2f}")
    lines.append(f"leave-one-year-out coverage of the {lv:.0%} interval: " + ", ".join(cov))
allres = pd.concat(res.values())
q = {lv: (allres.quantile((1 - lv) / 2), allres.quantile((1 + lv) / 2)) for lv in (0.8, 0.95)}
coef_all = fit(range(2022, 2027)); coef_full = fit(range(2022, 2026))
lines.insert(0, f"fit 2022-2026: a = {coef_all[0]:.3f}, b = {coef_all[1]:.3f};  fit 2022-2025 (full years only): "
                f"a = {coef_full[0]:.3f}, b = {coef_full[1]:.3f}")
lines.insert(1, f"out-of-sample residuals (n = {len(allres)}): 80% [{q[0.8][0]:.2f}, {q[0.8][1]:.2f}], "
                f"95% [{q[0.95][0]:.2f}, {q[0.95][1]:.2f}] logit")
a, b = coef_all
x = dM(2026); pred = a + b * x
pred_full = coef_full[0] + coef_full[1] * x
lines.append(f"Spearman of the 2027 forecasts, fit with vs without 2026: {spearmanr(pred, pred_full).statistic:.3f}")
z26 = Z[("robotics", 2026)]
out = pd.DataFrame({
    "topic": topics,
    "mlcv_2025": S[("mlcv", 2025)].values, "mlcv_2026": S[("mlcv", 2026)].values, "mlcv_change_logit": x.values,
    "robotics_2025": S[("robotics", 2025)].values, "robotics_2026": S[("robotics", 2026)].values,
    "robotics_2026_floored": (S[("robotics", 2026)] < 0.05).values,
    "pred_change_logit": pred.values,
    "robotics_2027": inv(z26 + pred).values,
    "lo80": inv(z26 + pred + q[0.8][0]).values, "hi80": inv(z26 + pred + q[0.8][1]).values,
    "lo95": inv(z26 + pred + q[0.95][0]).values, "hi95": inv(z26 + pred + q[0.95][1]).values,
    "pred_change_fit_2022_2025": pred_full.values,
})
out["rank"] = out.pred_change_logit.rank(ascending=False).astype(int)
out = out.sort_values("rank")
out.round(4).to_csv(f"{ROOT}/results/forecast_2027.csv", index=False)
open(f"{ROOT}/results/forecast_fit.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
pd.set_option("display.width", 200)
print(out[["rank", "topic", "mlcv_2025", "mlcv_2026", "robotics_2025", "robotics_2026", "robotics_2027", "lo80", "hi80"]]
      .round(2).to_string(index=False))
