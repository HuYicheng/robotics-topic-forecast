"""Forecast v2, step 10 (DESIGN_v2.md signal 3): ML/CV papers about robots = ML/CV rule "Robotics / embodied AI".
The robotics topic rules (~/robotics-trends/scripts/04_topics.py: title rule on the title, abstract rule on the
abstract) are applied to them; the ML/CV topics come from the ML/CV matrix.
Reading samples to check the robotics rules on this ML/CV text: topics with >= 20 matches; per period up to 6 title and
6 abstract-only matches, full abstract, matched words in [[ ]].
Output: data/embodied_matrix.csv, results/emb_review_samples/<period>/<topic>.txt, results/emb_review_samples/pools.csv
"""
import csv, importlib.util, json, os, random, re
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("rt", os.path.expanduser("~/robotics-trends/scripts/04_topics.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
MC = os.path.expanduser("~/mlcv-trends/data")
M = pd.read_csv(f"{MC}/topic_matrix_ta.csv"); M = M[M["Robotics / embodied AI"] == 1]
P = {r["uid"]: r for r in csv.DictReader(open(f"{MC}/papers.csv"))}
A = {}
for l in open(f"{MC}/abstracts.jsonl"):
    r = json.loads(l)
    if r["uid"] in set(M.uid): A[r["uid"]] = re.sub(r"https?://\S+", " ", r["abstract"] or "")
R_T = [(n, rx, rt.ABSTRACT_COMPILED[n]) for n, f, rx in rt.COMPILED if f != "E term"]
rows = []
for uid in M.uid:
    ti, ab = P[uid]["title"], A.get(uid, "")
    rows.append([uid] + [int(bool(trx.search(ti) or arx.search(ab))) for _, trx, arx in R_T])
E = pd.DataFrame(rows, columns=["uid"] + ["R: " + n for n, _, _ in R_T])
mcols = [c for c in M.columns if c not in ("uid", "venue", "year") and not c.startswith("[term]")]
E = M[["uid", "venue", "year"] + mcols].rename(columns={c: "M: " + c for c in mcols}).merge(E, on="uid")
E.to_csv(f"{ROOT}/data/embodied_matrix.csv", index=False)
print(len(E), "embodied ML/CV papers;", E.groupby("year").size().to_dict())
mark = lambda rx, s: rx.sub(lambda m: "[[" + m.group(0) + "]]", s)
out = f"{ROOT}/results/emb_review_samples"; pools = []
for n, trx, arx in R_T:
    tot = int(E["R: " + n].sum())
    if tot < 20: pools.append([n, "all", tot, 0, 0]); continue
    safe = re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")
    for per, keep in (("2020-2023", lambda y: y <= 2023), ("2024-2026", lambda y: y >= 2024)):
        ps = [u for u, y in zip(E.uid, E.year) if keep(y)]
        T = [u for u in ps if trx.search(P[u]["title"])]
        Ao = [u for u in ps if not trx.search(P[u]["title"]) and arx.search(A.get(u, ""))]
        pools.append([n, per, len(ps), len(T), len(Ao)])
        rnd = random.Random(f"0|{n}|{per}"); os.makedirs(f"{out}/{per}", exist_ok=True)
        with open(f"{out}/{per}/{safe}.txt", "w") as f:
            f.write(f"# {n} | {per} | ML/CV embodied papers {len(ps)} | title matches {len(T)} | abstract-only matches {len(Ao)}\n\n")
            for i, u in enumerate(rnd.sample(T, min(6, len(T))), 1):
                f.write(f"[T{i}] {u} | {P[u]['venue']} {P[u]['year']} | {mark(trx, P[u]['title'])}\n     abstract: {mark(arx, A.get(u, ''))}\n\n")
            for i, u in enumerate(rnd.sample(Ao, min(6, len(Ao))), 1):
                f.write(f"[A{i}] {u} | {P[u]['venue']} {P[u]['year']} | {P[u]['title']}\n     abstract: {mark(arx, A.get(u, ''))}\n\n")
pd.DataFrame(pools, columns=["topic", "period", "papers", "title_matches", "abstract_only_matches"]).to_csv(f"{out}/pools.csv", index=False)
