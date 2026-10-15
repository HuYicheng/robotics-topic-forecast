"""Forecast v2, step 11: the series behind the signals in DESIGN_v2.md.
- data/v2_topics.csv      : topic set (R topics + M topics not duplicated by an R topic; Jaccard >= 0.6 on robotics papers)
- data/v2_rob_series.csv  : robotics standardised share per topic and year (venue + year grouped binomial GLM, 2025 venue
                            mix, 95% interval from 2000 coefficient draws), plus raw counts at the early venues
                            (CoRL + RSS) and the rest
- data/v2_emb_series.csv  : ML/CV papers about robots, standardised share (2024 + 2025 venue mix of that subset)
- data/v2_mlcv_split.csv  : M topics in all ML/CV papers, standardised share of papers that name the topic AND
                            (part "llm") name LLM / VLM / reasoning, or (part "other") do not; denominator all papers
"""
import os, warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
YEARS = list(range(2020, 2027)); EARLY = {"CoRL", "RSS"}
def std_share(df, col, ref, draws=0, rng=None):
    d = df[["venue", "year", col]].rename(columns={col: "y"})
    keep = d.groupby("venue").y.sum(); keep = keep[keep > 0].index
    out = {}
    if len(keep) == 0:
        return {y: (0.0, 0.0, 0.0) for y in YEARS}
    g = d[d.venue.isin(keep)].groupby(["venue", "year"], as_index=False).agg(k=("y", "sum"), n=("y", "size")); g["nk"] = g.n - g.k
    f = smf.glm("k + nk ~ C(venue) + C(year)", data=g, family=sm.families.Binomial()).fit()
    names = list(f.params.index)
    D = rng.multivariate_normal(f.params.values, f.cov_params().values, size=draws) if draws else None
    for y in YEARS:
        X, w = [], []
        for v, s in ref.items():
            if v not in keep: continue
            x = np.zeros(len(names)); x[0] = 1
            for nm in (f"C(venue)[T.{v}]", f"C(year)[T.{y}]"):
                if nm in names: x[names.index(nm)] = 1
            X.append(x); w.append(s)
        X, w = np.array(X), np.array(w)
        est = 100 * float((w / (1 + np.exp(-X @ f.params.values))).sum())
        if draws:
            sims = 100 * ((1 / (1 + np.exp(-D @ X.T))) * w).sum(axis=1); out[y] = (est, np.percentile(sims, 2.5), np.percentile(sims, 97.5))
        else: out[y] = (est, np.nan, np.nan)
    return out
rng = np.random.default_rng(0)
Rm = pd.read_csv(os.path.expanduser("~/robotics-trends/data/topic_matrix_ta.csv"))
Rcols = [c for c in Rm.columns if c not in ("uid", "venue", "year") and not c.startswith("[term]")]
Mm = pd.read_csv(f"{ROOT}/data/robotics_mlcv_matrix.csv")
ok = [l.strip() for l in open(f"{ROOT}/data/topics_ok.txt")]
J = Rm[["uid"] + Rcols].merge(Mm[["uid"] + ok], on="uid", suffixes=("", " (M)"))
dup = []
for m in ok:
    a = J[m + " (M)" if m in Rcols else m].values.astype(bool)
    for r in Rcols:
        b = J[r].values.astype(bool); jac = (a & b).sum() / max((a | b).sum(), 1)
        if jac >= 0.6: dup.append([m, r, round(jac, 3)])
dup = pd.DataFrame(dup, columns=["mlcv_topic", "robotics_topic", "jaccard"]); dup.to_csv(f"{ROOT}/results/v2_duplicates.csv", index=False)
print(dup.to_string(index=False))
Mkeep = [m for m in ok if m not in set(dup.mlcv_topic)]
topics = pd.DataFrame([[r, "R", r] for r in Rcols] + [[m, "M", m] for m in Mkeep], columns=["topic", "set", "rule"])
topics["key"] = topics.set + ": " + topics.topic; topics.to_csv(f"{ROOT}/data/v2_topics.csv", index=False)
print(len(Rcols), "R topics,", len(Mkeep), "M topics kept")
# robotics series
R = Rm[["uid", "venue", "year"] + Rcols].rename(columns={c: "R: " + c for c in Rcols}).merge(
    Mm[["uid"] + Mkeep].rename(columns={c: "M: " + c for c in Mkeep}), on="uid")
ref = R[R.year == 2025].venue.value_counts(normalize=True); rows = []
for key in topics.key:
    s = std_share(R, key, ref, 2000, rng)
    for y in YEARS:
        d = R[R.year == y]; e = d.venue.isin(EARLY)
        rows.append([key, y, *s[y], int(d[key].sum()), len(d), int(d.loc[e, key].sum()), int(e.sum()), int(d.loc[~e, key].sum()), int((~e).sum())])
pd.DataFrame(rows, columns=["key", "year", "share", "lo", "hi", "k", "n", "k_early", "n_early", "k_rest", "n_rest"]).to_csv(f"{ROOT}/data/v2_rob_series.csv", index=False)
# embodied ML/CV series
E = pd.read_csv(f"{ROOT}/data/embodied_matrix.csv"); refE = E[E.year.isin([2024, 2025])].venue.value_counts(normalize=True); rows = []
for key in topics.key:
    if key not in E.columns: continue
    s = std_share(E, key, refE)
    for y in YEARS: rows.append([key, y, s[y][0], int(E.loc[E.year == y, key].sum()), int((E.year == y).sum())])
pd.DataFrame(rows, columns=["key", "year", "share", "k", "n"]).to_csv(f"{ROOT}/data/v2_emb_series.csv", index=False)
# ML/CV split by LLM / VLM / reasoning
A = pd.read_csv(os.path.expanduser("~/mlcv-trends/data/topic_matrix_ta.csv"))
llm = A[["LLMs / language models", "Vision-language / multimodal LLMs", "Reasoning / chain-of-thought"]].max(axis=1) == 1
refA = A[A.year.isin([2024, 2025])].venue.value_counts(normalize=True); rows = []
for m in ok:
    for part, mask in (("llm", llm), ("other", ~llm), ("all", llm | ~llm)):
        A["_x"] = (A[m] == 1) & mask; s = std_share(A, "_x", refA)
        for y in YEARS: rows.append(["M: " + m, part, y, s[y][0], int(A.loc[A.year == y, "_x"].sum()), int((A.year == y).sum())])
pd.DataFrame(rows, columns=["key", "part", "year", "share", "k", "n"]).to_csv(f"{ROOT}/data/v2_mlcv_split.csv", index=False)
print("done")
