"""Forecast v2, step 13 (DESIGN_v2.md signal 6): emerging word n-grams not covered by any topic rule.
Corpus: titles + abstracts of robotics papers and ML/CV papers about robots. n-grams of 1-3 words (lower case, letters
and digits, hyphens kept), not starting or ending with a stop word, counted once per paper. An n-gram is "covered" when
any robotics or ML/CV title rule matches the n-gram itself.
Emergence at year Y: log((k_Y + 1)/(n_Y + 2)) - log((k_{Y-2..Y-1} + 1)/(n_{Y-2..Y-1} + 2)), k_Y >= 10 papers.
Backtest, Y = T-1 for T = 2024, 2025, 2026: top 30 uncovered n-grams (an n-gram is dropped when a longer selected one
contains it) vs 300 random uncovered n-grams with k_Y in the same frequency band (±50%); outcome = change of the robotics
paper share T-1 -> T (log of (k+1)/(n+2)). Reported: median outcome and share with outcome > 0, both groups.
Current list: Y = 2026, top 40, with counts per year (read from abstracts before any is called a topic).
Output: results/v2_emerging_backtest.csv, results/v2_emerging_2026.csv
"""
import os, re, csv, json, collections, importlib.util, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
def load(p):
    s = importlib.util.spec_from_file_location("m" + str(abs(hash(p))), p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rt = load(os.path.expanduser("~/robotics-trends/scripts/04_topics.py")); mt = load(os.path.expanduser("~/mlcv-trends/scripts/04_topics.py"))
RX = [rx for _, f, rx in rt.COMPILED if f != "E term"] + [rx for _, f, rx in mt.COMPILED if not str(f).startswith("E")]
STOP = set("a an the of for and or in on to with by from via at as is are be we our this that these those its it into using based than their can which while has have not more new two one under over between across toward towards through".split())
docs = []  # (field, year, text)
RD = os.path.expanduser("~/robotics-trends/data")
rp = {r["uid"]: r for r in csv.DictReader(open(f"{RD}/papers.csv"))}
for l in open(f"{RD}/abstracts.jsonl"):
    r = json.loads(l); p = rp[r["uid"]]; docs.append(("rob", int(p["year"]), p["title"] + " . " + (r["abstract"] or "")))
E = set(pd.read_csv(f"{ROOT}/data/embodied_matrix.csv").uid)
MD = os.path.expanduser("~/mlcv-trends/data"); mp = {r["uid"]: r for r in csv.DictReader(open(f"{MD}/papers.csv"))}
for l in open(f"{MD}/abstracts.jsonl"):
    r = json.loads(l)
    if r["uid"] in E: p = mp[r["uid"]]; docs.append(("emb", int(p["year"]), p["title"] + " . " + (r["abstract"] or "")))
tok = re.compile(r"[a-z0-9][a-z0-9-]*")
def grams(t):
    out = set()
    for sent in re.split(r"[.;:!?()\[\]]", re.sub(r"https?://\S+", " ", t.lower())):
        w = tok.findall(sent)
        for n in (1, 2, 3):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if g[0] in STOP or g[-1] in STOP or g[0].isdigit() or len(g[0]) < 3: continue
                out.add(" ".join(g))
    return out
cnt = {f: collections.defaultdict(collections.Counter) for f in ("rob", "both")}
N = {f: collections.Counter() for f in ("rob", "both")}
for f, y, t in docs:
    g = grams(t)
    cnt["both"][y].update(g); N["both"][y] += 1
    if f == "rob": cnt["rob"][y].update(g); N["rob"][y] += 1
covered = {}
def is_cov(g):
    if g not in covered: covered[g] = any(rx.search(g) for rx in RX)
    return covered[g]
lr = lambda k, n: np.log((k + 1) / (n + 2))
def emergence(Y):
    c, n = cnt["both"], N["both"]
    prev = c[Y - 1] + c[Y - 2]; nprev = n[Y - 1] + n[Y - 2]
    rows = [(g, k, lr(k, n[Y]) - lr(prev.get(g, 0), nprev)) for g, k in c[Y].items() if k >= 10]
    return pd.DataFrame(rows, columns=["gram", "k", "em"])
def top(df, m):
    sel = []
    for g in df.sort_values("em", ascending=False).gram:
        if is_cov(g) or any(g in s for s in sel): continue
        sel.append(g)
        if len(sel) == m: break
    return sel
rng = np.random.default_rng(0); out = []
for T in (2024, 2025, 2026):
    Y = T - 1; df = emergence(Y); sel = top(df, 30); ks = df.set_index("gram").k
    outc = lambda g: lr(cnt["rob"][T].get(g, 0), N["rob"][T]) - lr(cnt["rob"][Y].get(g, 0), N["rob"][Y])
    pool = [g for g in df.gram if not is_cov(g) and g not in sel]
    rnd = []
    for g in sel:
        band = [h for h in pool if 0.5 * ks[g] <= ks[h] <= 1.5 * ks[g]]
        rnd += list(rng.choice(band, size=min(10, len(band)), replace=False))
    a, b = np.array([outc(g) for g in sel]), np.array([outc(g) for g in rnd])
    out.append([T, len(a), round(np.median(a), 3), round((a > 0).mean(), 3), len(b), round(np.median(b), 3), round((b > 0).mean(), 3), "; ".join(sel[:12])])
bt = pd.DataFrame(out, columns=["target_year", "n_top", "median_change_top", "share_up_top", "n_random", "median_change_random", "share_up_random", "first_12_top"])
bt.to_csv(f"{ROOT}/results/v2_emerging_backtest.csv", index=False); print(bt.drop(columns="first_12_top").to_string(index=False))
df = emergence(2026); sel = top(df, 40)
cur = pd.DataFrame([[g] + [cnt["rob"][y].get(g, 0) for y in range(2020, 2027)] + [cnt["both"][y].get(g, 0) - cnt["rob"][y].get(g, 0) for y in range(2020, 2027)]
                    for g in sel], columns=["gram"] + [f"rob_{y}" for y in range(2020, 2027)] + [f"emb_{y}" for y in range(2020, 2027)])
cur.to_csv(f"{ROOT}/results/v2_emerging_2026.csv", index=False); print(cur.to_string(index=False))
