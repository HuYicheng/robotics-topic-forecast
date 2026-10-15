"""Robotics forecast, step 3: standardised share per year of every ML / CV topic in both fields, same rule in both.
ML / CV: ~/mlcv-trends/results/shares_and_change_ta.csv (venue mix of 2024 + 2025, model topic ~ venue + year).
Robotics: the same model on data/robotics_mlcv_matrix.csv, venue mix of 2025 (as in the robotics study); IROS / CoRL 2026
not published, so robotics 2026 is estimated from the other venues. Fitted on venue-year counts (grouped binomial GLM).
Output: data/series.csv (topic, field, year, share, lo, hi, k, n)
"""
import os, warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
YEARS = list(range(2020, 2027))
r = pd.read_csv(f"{ROOT}/data/robotics_mlcv_matrix.csv")
topics = [c for c in r.columns if c not in ("uid", "venue", "year")]
ref = r[r.year == 2025].venue.value_counts(normalize=True)
rng = np.random.default_rng(0); rows = []
for tp in topics:
    d = r[["venue", "year", tp]].rename(columns={tp: "y"})
    keep = d.groupby("venue").y.sum(); keep = keep[keep > 0].index
    g = d[d.venue.isin(keep)].groupby(["venue", "year"], as_index=False).agg(k=("y", "sum"), n=("y", "size"))
    g["nk"] = g.n - g.k
    if len(keep) == 0 or g.k.sum() == 0:
        for y in YEARS: rows.append([tp, "robotics", y, 0.0, 0.0, 0.0, 0, int((r.year == y).sum())])
        continue
    f = smf.glm("k + nk ~ C(venue) + C(year)", data=g, family=sm.families.Binomial()).fit()
    names = list(f.params.index); draws = rng.multivariate_normal(f.params.values, f.cov_params().values, size=2000)
    for y in YEARS:
        X, w = [], []
        for v, s in ref.items():
            if v not in keep: continue
            x = np.zeros(len(names)); x[0] = 1
            if f"C(venue)[T.{v}]" in names: x[names.index(f"C(venue)[T.{v}]")] = 1
            if f"C(year)[T.{y}]" in names: x[names.index(f"C(year)[T.{y}]")] = 1
            X.append(x); w.append(s)
        X, w = np.array(X), np.array(w)
        est = 100 * float((w / (1 + np.exp(-X @ f.params.values))).sum())
        sims = 100 * ((1 / (1 + np.exp(-draws @ X.T))) * w).sum(axis=1)
        rows.append([tp, "robotics", y, est, np.percentile(sims, 2.5), np.percentile(sims, 97.5), int(d[d.year == y].y.sum()), int((d.year == y).sum())])
m = pd.read_csv(os.path.expanduser("~/mlcv-trends/results/shares_and_change_ta.csv"))
for _, x in m.iterrows():
    for y in YEARS:
        rows.append([x.topic, "mlcv", y, x[f"share_{y}"], x[f"share_{y}_lo"], x[f"share_{y}_hi"], int(x[f"n_{y}"]), None])
pd.DataFrame(rows, columns=["topic", "field", "year", "share", "lo", "hi", "k", "n"]).to_csv(f"{ROOT}/data/series.csv", index=False)
print("series written for", len(topics), "topics")
