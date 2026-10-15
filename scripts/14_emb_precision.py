"""Forecast v2, step 14: precision of the robotics rules on ML/CV papers about robots, from the abstract-judged review
(scratchpad emb_review/E*.json, copied to results/emb_review_labels/). Per topic and period: pool-weighted mix of title
and abstract-only items; correct = Y + P (primary) or Y only; "?" excluded (DESIGN_v2.md).
Output: results/emb_precision.csv
"""
import os, re, json, glob, shutil, importlib.util, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = "/tmp/claude-1000/-home-yicheng/76ef451b-35a8-40dc-b27a-11fc0c9cc4ce/scratchpad/emb_review"
os.makedirs(f"{ROOT}/results/emb_review_labels", exist_ok=True)
for f in glob.glob(f"{SRC}/E*.json"): shutil.copy(f, f"{ROOT}/results/emb_review_labels/")
pools = pd.read_csv(f"{ROOT}/results/emb_review_samples/pools.csv")
spec = importlib.util.spec_from_file_location("rt", os.path.expanduser("~/robotics-trends/scripts/04_topics.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
stem = {re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_"): n for n, f, _ in rt.COMPILED}
lab = {}
for f in sorted(glob.glob(f"{ROOT}/results/emb_review_labels/E*.json")):
    for t in json.load(open(f))["topics"]: lab[(stem[t["topic"]], t["period"])] = t
def prec(name, per, ok):
    t = lab.get((name, per)); p = pools[(pools.topic == name) & (pools.period == per)]
    if t is None or p.empty: return None, 0
    nT, nA = int(p.title_matches.iloc[0]), int(p.abstract_only_matches.iloc[0]); num = den = 0
    for part, n in (("T", nT), ("A", nA)):
        c = t.get(part) or {}; judged = c.get("read", 0) - c.get("?", 0)
        if judged > 0 and n > 0: num += n * sum(c.get(k, 0) for k in ok) / judged; den += n
    return (num / den if den else None), nT + nA
rows = []
for name in pools.topic.unique():
    r = [name]
    for per in ("2020-2023", "2024-2026"):
        a, n = prec(name, per, "YP"); b, _ = prec(name, per, "Y"); r += [a, b, n]
    rows.append(r)
out = pd.DataFrame(rows, columns=["topic", "prec_2020", "precY_2020", "pool_2020", "prec_2024", "precY_2024", "pool_2024"]).round(3)
out.to_csv(f"{ROOT}/results/emb_precision.csv", index=False); print(out.to_string(index=False))
