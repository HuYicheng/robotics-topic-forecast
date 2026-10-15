"""Forecast v2, step 12: signals 1-5 of DESIGN_v2.md, backtest and 2027 forecast.
Inputs: data/v2_topics.csv, data/v2_rob_series.csv, data/v2_emb_series.csv, data/v2_mlcv_split.csv,
results/v2_duplicates.csv, results/emb_precision.csv (rule check of signal 3), the paper x topic matrices.
Outputs: results/v2_signals.csv (every signal, topic, target year), v2_single.csv, v2_backtest.csv, v2_ablation.csv,
v2_bootstrap.csv, v2_forecast_2027.csv, v2_fit.txt
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
lg = lambda p: np.log(np.clip(p, 0.05, None) / (100 - np.clip(p, 0.05, None)))
lgc = lambda k, n: np.log((k + 0.5) / (n - k + 0.5))
TP = pd.read_csv(f"{ROOT}/data/v2_topics.csv"); KEYS = list(TP.key); SET = dict(zip(TP.key, TP.set))
RS = pd.read_csv(f"{ROOT}/data/v2_rob_series.csv")
Z = RS.pivot(index="key", columns="year", values="share").loc[KEYS].apply(lg)
GE = (lgc(RS.k_early, RS.n_early) - lgc(RS.k_rest, RS.n_rest)); RS["g"] = GE
G = RS.pivot(index="key", columns="year", values="g").loc[KEYS]
ES = pd.read_csv(f"{ROOT}/data/v2_emb_series.csv")
ZE = ES.pivot(index="key", columns="year", values="share").apply(lg)
EK = ES.groupby("key").k.sum()
prec = pd.read_csv(f"{ROOT}/results/emb_precision.csv").set_index("topic")
def emb_ok(key):
    if EK.get(key, 0) < 20: return False
    if key.startswith("M: "): return True            # ML/CV rules on ML/CV text: checked in the ML/CV study
    t = key[3:]
    if t not in prec.index: return False
    p = prec.loc[t]
    return p.prec_2024 >= 0.8 and (p.prec_2020 >= 0.8 or p.pool_2020 < 20)
EMB_OK = {k: emb_ok(k) for k in KEYS}
# ML/CV split series and the topic each key takes it from
MS = pd.read_csv(f"{ROOT}/data/v2_mlcv_split.csv")
ZM = {part: MS[MS.part == part].pivot(index="key", columns="year", values="share").apply(lg) for part in ("llm", "other", "all")}
dup = pd.read_csv(f"{ROOT}/results/v2_duplicates.csv").sort_values("jaccard", ascending=False).drop_duplicates("robotics_topic")
MLKEY = {k: k for k in KEYS if k.startswith("M: ")}
MLKEY.update({"R: " + r: "M: " + m for m, r in zip(dup.mlcv_topic, dup.robotics_topic)})
DEFINING = {"M: LLMs / language models", "M: Vision-language / multimodal LLMs", "M: Reasoning / chain-of-thought"}
# matrices for co-topic profiles (ML/CV rule columns in both fields)
MC = pd.read_csv(os.path.expanduser("~/mlcv-trends/data/topic_matrix_ta.csv"))
RB = pd.read_csv(f"{ROOT}/data/robotics_mlcv_matrix.csv")
RR = pd.read_csv(os.path.expanduser("~/robotics-trends/data/topic_matrix_ta.csv"))
RB = RB.merge(RR[["uid"] + [k[3:] for k in KEYS if k.startswith("R: ")]].rename(columns=lambda c: c if c == "uid" else "R: " + c), on="uid")
for k in KEYS:
    if k.startswith("M: "): RB[k] = RB[k[3:]]
PROF = [c for c in MC.columns if c not in ("uid", "venue", "year", "Robotics / embodied AI", "LLMs / language models",
        "Vision-language / multimodal LLMs", "Reasoning / chain-of-thought") and not c.startswith("[term]")]
LLM = MC[["LLMs / language models", "Vision-language / multimodal LLMs", "Reasoning / chain-of-thought"]].max(axis=1) == 1
def loglift(df, mask, cols):
    sub = df.loc[mask, cols]; base = df[cols].mean()
    return np.log((sub.sum() + 0.5) / (len(sub) + 1)) - np.log(base + 1e-4)
def chosen_part(key, T):
    mk = MLKEY[key]
    if mk in DEFINING: return "all"
    yrs = MC.year.between(T - 3, T - 1); ry = RB.year.between(T - 3, T - 1)
    rcol = key if key in RB.columns else key[3:]
    rprof = loglift(RB[ry], RB.loc[ry, rcol] == 1, PROF)
    best, bv = "other", -9
    for part, pm in (("llm", LLM), ("other", ~LLM)):
        m = (MC.loc[yrs, mk[3:]] == 1) & pm[yrs]
        if m.sum() < 10: continue
        cols = [c for c in PROF if c != mk[3:]]
        v = spearmanr(loglift(MC[yrs], m, cols), rprof[cols]).statistic
        if v > bv: best, bv = part, v
    return best
METHODS = [k for k in KEYS if k.startswith("M: ")] + ["R: " + n for n in TP[TP.set == "R"].topic
           if n in set(pd.read_csv(os.path.expanduser("~/robotics-trends/data/topic_matrix_ta.csv"), nrows=0).columns)]
import importlib.util
spec = importlib.util.spec_from_file_location("rt", os.path.expanduser("~/robotics-trends/scripts/04_topics.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
LEARN = {"R: " + n for n, f, _ in rt.TOPICS if f == "A learning"}
METHODS = [k for k in KEYS if k.startswith("M: ") or k in LEARN]
def signals(T):
    yrs = list(range(2020, T)); x = np.array(yrs) - np.mean(yrs)
    S = pd.DataFrame(index=KEYS)
    S["own"] = Z[yrs].values @ x / (x @ x)
    S["level"] = Z[T - 1]
    S["venue"] = G[T - 1] - G[list(range(2020, T - 1))].mean(axis=1)
    S["emb"] = [(ZE.at[k, T - 1] - ZE.at[k, T - 2]) if EMB_OK[k] and k in ZE.index else np.nan for k in KEYS]
    dm = Z[T - 1] - Z[T - 2]; d = RB[RB.year == T - 1]
    S["diff"] = [sum(d.loc[d[k] == 1, m].mean() * dm[m] for m in METHODS if m != k) if (d[k] == 1).sum() else np.nan for k in KEYS]
    parts, ml = {}, []
    for k in KEYS:
        if k not in MLKEY: ml.append(np.nan); continue
        p = chosen_part(k, T); parts[k] = p; mk = MLKEY[k]
        ml.append(ZM[p].at[mk, T - 1] - ZM[p].at[mk, T - 2])
    S["mlcv"] = ml; S["mlcv_part"] = pd.Series(parts)
    return S
SIG = ["own", "level", "venue", "emb", "diff", "mlcv"]; SIGN = {"level": -1}
def score(S, use=SIG):
    zs = pd.DataFrame({s: SIGN.get(s, 1) * (S[s] - S[s].mean()) / S[s].std() for s in use})
    return zs.mean(axis=1), zs
TARGETS = [2023, 2024, 2025, 2026]
S_T = {T: signals(T) for T in TARGETS + [2027]}
dR = {T: Z[T] - Z[T - 1] for T in TARGETS}
allsig = pd.concat([S_T[T].assign(target_year=T) for T in S_T]); allsig.round(4).to_csv(f"{ROOT}/results/v2_signals.csv")
def calfit(tr, use=SIG):
    x = pd.concat([score(S_T[t], use)[0] for t in tr]); y = pd.concat([dR[t] for t in tr])
    c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x.values]), y.values, rcond=None); return c
def ols_fit(tr):
    X = pd.concat([score(S_T[t])[1].fillna(0) for t in tr]); y = pd.concat([dR[t] for t in tr])
    c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X.values]), y.values, rcond=None); return c
def metrics(p, a, sub):
    p, a = p.loc[sub], a.loc[sub]
    return round(spearmanr(p, a).statistic, 3), round(float((p - a).abs().mean()), 3), len(set(p.nlargest(10).index) & set(a.nlargest(10).index))
subsets = {"all": KEYS, "R": [k for k in KEYS if SET[k] == "R"], "M": [k for k in KEYS if SET[k] == "M"]}
rows, single, abl, preds, resid = [], [], [], {}, []
for T in (2024, 2025, 2026):
    S, a = S_T[T], dR[T]; tr = [t for t in TARGETS if t < T]
    c = calfit(tr); sc = score(S)[0]; p_score = c[0] + c[1] * sc
    w = ols_fit(tr); Xz = score(S)[1].fillna(0); p_ols = pd.Series(np.column_stack([np.ones(len(Xz)), Xz.values]) @ w, index=KEYS)
    p_own = S["own"]
    preds[T] = {"score": p_score, "ols": p_ols, "own": p_own}
    resid.append(a - p_score)
    for name, p in preds[T].items():
        for sname, sub in subsets.items(): rows.append([T, name, sname, *metrics(p, a, sub)])
    for s in SIG:
        v = SIGN.get(s, 1) * S[s]; m = v.notna()
        single.append([T, s, int(m.sum()), round(spearmanr(v[m], a[m]).statistic, 3)])
    for s in SIG:
        use = [u for u in SIG if u != s]; cc = calfit(tr, use); p = cc[0] + cc[1] * score(S, use)[0]
        abl.append([T, s, metrics(p, a, KEYS)[0], metrics(p_score, a, KEYS)[0]])
cols = ["target_year", "model", "topics", "spearman", "mae_logit", "top10_overlap"]
bt = pd.DataFrame(rows, columns=cols); bt.to_csv(f"{ROOT}/results/v2_backtest.csv", index=False)
sg = pd.DataFrame(single, columns=["target_year", "signal", "n_topics", "spearman_with_actual"]); sg.to_csv(f"{ROOT}/results/v2_single.csv", index=False)
ab = pd.DataFrame(abl, columns=["target_year", "left_out", "spearman_without", "spearman_full"]); ab.to_csv(f"{ROOT}/results/v2_ablation.csv", index=False)
rng = np.random.default_rng(0); bs = []
for T in (2024, 2025, 2026):
    A = dR[T].values; P1 = preds[T]["score"].values; P0 = preds[T]["own"].values
    d = [spearmanr(P1[i], A[i]).statistic - spearmanr(P0[i], A[i]).statistic for i in (rng.integers(0, len(A), len(A)) for _ in range(2000))]
    bs.append([T, round(spearmanr(P1, A).statistic - spearmanr(P0, A).statistic, 3), *np.round(np.nanquantile(d, [0.025, 0.975]), 3)])
bs = pd.DataFrame(bs, columns=["target_year", "rho_score_minus_own", "lo95", "hi95"]); bs.to_csv(f"{ROOT}/results/v2_bootstrap.csv", index=False)
# forecast 2027
c = calfit(TARGETS); S = S_T[2027]; sc, zs = score(S); pred = c[0] + c[1] * sc
inv = lambda z: 100 / (1 + np.exp(-z)); base = Z[2026]
band = lambda share: pd.cut(share, [0, 1, 3, 100], labels=["<1%", "1-3%", ">=3%"])
rb = pd.concat([pd.DataFrame({"r": r, "b": band(inv(Z[T - 1]))}) for T, r in zip((2024, 2025, 2026), resid)])
res = rb.r
Q = rb.groupby("b", observed=True).r.quantile([0.1, 0.9, 0.025, 0.975]).unstack()
b26 = band(inv(base)); lo80_, hi80_, lo95_, hi95_ = (b26.map(Q[c]).astype(float) for c in (0.1, 0.9, 0.025, 0.975))
q80, q95 = res.quantile([0.1, 0.9]).values, res.quantile([0.025, 0.975]).values
fc = pd.DataFrame({"key": KEYS, "set": [SET[k] for k in KEYS], "share_2025": inv(Z[2025]).values, "share_2026": inv(base).values,
    "pred_change": pred.values, "share_2027": inv(base + pred).values,
    "lo80": inv(base + pred + lo80_).values, "hi80": inv(base + pred + hi80_).values,
    "lo95": inv(base + pred + lo95_).values, "hi95": inv(base + pred + hi95_).values, "size_band": b26.astype(str).values,
    "n_signals": zs.notna().sum(axis=1).values, "mlcv_part": S.mlcv_part.reindex(KEYS).values})
for s in SIG: fc["z_" + s] = zs[s].values
fc["disagree"] = (zs.gt(0.5).any(axis=1) & zs.lt(-0.5).any(axis=1)).values
fc = fc.sort_values("pred_change", ascending=False); fc.insert(0, "rank", range(1, len(fc) + 1))
fc.round(4).to_csv(f"{ROOT}/results/v2_forecast_2027.csv", index=False)
cov = [f"{T}: 80% {((r >= q80[0]) & (r <= q80[1])).mean():.2f}" for T, r in zip((2024, 2025, 2026), resid)]
lines = [f"score calibration on 2023-2026: a = {c[0]:.3f}, b = {c[1]:.3f}",
         f"residuals (out-of-sample 2024-2026, n = {len(res)}): 80% [{q80[0]:.2f}, {q80[1]:.2f}], 95% [{q95[0]:.2f}, {q95[1]:.2f}]",
         "error quantiles by size band (10/90/2.5/97.5%): " + "; ".join(f"{b}: " + ", ".join(f"{v:.2f}" for v in Q.loc[b]) for b in Q.index),
         "in-sample coverage of the pooled 80% interval per year: " + ", ".join(cov),
         "emb signal used for: " + ", ".join(k for k in KEYS if EMB_OK[k])]
open(f"{ROOT}/results/v2_fit.txt", "w").write("\n".join(lines) + "\n")
pd.set_option("display.width", 220)
print(bt.to_string(index=False)); print(sg.pivot(index="signal", columns="target_year", values="spearman_with_actual"))
print(ab.to_string(index=False)); print(bs.to_string(index=False)); print("\n".join(lines))
