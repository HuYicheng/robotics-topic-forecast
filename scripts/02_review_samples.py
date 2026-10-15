"""Robotics forecast, step 2: reading samples to check the ML / CV topic rules on robotics text (a word may have another
sense in robotics). For each topic with at least 20 robotics papers in 2020-2026: per period (2020-2023, 2024-2026) up to
N random papers matched by the title rule and up to N matched only by the abstract rule, with the full abstract.
Output: results/review_samples/<period>/<topic>.txt, results/review_samples/pools.csv
"""
import csv, importlib.util, json, os, random, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/mlcv-trends/scripts/04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
R = os.path.expanduser("~/robotics-trends/data")
papers = list(csv.DictReader(open(f"{R}/papers.csv")))
A = {r["uid"]: t.strip_urls(r["abstract"]) for r in map(json.loads, open(f"{R}/abstracts.jsonl"))}
N, SEED = 6, int(os.environ.get("SAMPLE_SEED", "0"))
mark = lambda rx, s: rx.sub(lambda m: "[[" + m.group(0) + "]]", s)
out = f"{ROOT}/results/review_samples"; pools = []
for name, fam, trx in t.COMPILED:
    if fam == "E term": continue
    arx = t.ABSTRACT_COMPILED[name]; safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    allm = [p for p in papers if p["uid"] in A and (trx.search(p["title"]) or arx.search(A[p["uid"]]))]
    if len(allm) < 20: continue
    for per, keep in (("2020-2023", lambda y: y <= 2023), ("2024-2026", lambda y: y >= 2024)):
        ps = [p for p in papers if p["uid"] in A and keep(int(p["year"]))]
        T = [p for p in ps if trx.search(p["title"])]
        Ao = [p for p in ps if not trx.search(p["title"]) and arx.search(A[p["uid"]])]
        pools.append([name, per, len(ps), len(T), len(Ao)])
        rnd = random.Random(f"{SEED}|{name}|{per}"); os.makedirs(f"{out}/{per}", exist_ok=True)
        with open(f"{out}/{per}/{safe}.txt", "w") as f:
            f.write(f"# {name} | {per} | robotics papers {len(ps)} | title matches {len(T)} | abstract-only matches {len(Ao)}\n\n")
            for i, p in enumerate(rnd.sample(T, min(N, len(T))), 1):
                f.write(f"[T{i}] {p['uid']} | {p['venue']} {p['year']} | {mark(trx, p['title'])}\n     abstract: {arx.mark(A[p['uid']])}\n")
            f.write("\n")
            for i, p in enumerate(rnd.sample(Ao, min(N, len(Ao))), 1):
                f.write(f"[A{i}] {p['uid']} | {p['venue']} {p['year']} | {p['title']}\n     abstract: {arx.mark(A[p['uid']])}\n")
with open(f"{out}/pools.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["topic", "period", "papers", "title_matches", "abstract_only_matches"]); w.writerows(pools)
print(len(pools) // 2, "topics sampled")
