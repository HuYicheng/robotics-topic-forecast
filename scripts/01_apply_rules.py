"""Robotics forecast, step 1: apply the ML / CV topic rules (~/mlcv-trends/scripts/04_topics.py, unchanged) to the
robotics research articles (~/robotics-trends/data, 2020-2026), so that each topic is measured by one and the same
rule in both fields (user, 2026-10-15: ML / CV is ahead of robotics; forecast next year's robotics topics).
Output: data/robotics_mlcv_matrix.csv (uid, venue, year, one 0/1 column per ML / CV topic; title rule OR abstract rule)
"""
import csv, importlib.util, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/mlcv-trends/scripts/04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
R = os.path.expanduser("~/robotics-trends/data")
papers = {p["uid"]: p for p in csv.DictReader(open(f"{R}/papers.csv"))}
abstracts = {r["uid"]: r["abstract"] for r in map(json.loads, open(f"{R}/abstracts.jsonl"))}
names = [n for n, f, _ in t.COMPILED if f != "E term"]
title_rx = {n: rx for n, _, rx in t.COMPILED}
with open(f"{ROOT}/data/robotics_mlcv_matrix.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["uid", "venue", "year"] + names)
    for uid, a in abstracts.items():
        p = papers[uid]; a = t.strip_urls(a)
        w.writerow([uid, p["venue"], p["year"]] + [int(bool(title_rx[n].search(p["title"]) or t.ABSTRACT_COMPILED[n].search(a))) for n in names])
print(len(abstracts), "robotics papers x", len(names), "ML/CV topics")
