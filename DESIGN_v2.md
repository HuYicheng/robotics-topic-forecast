# Forecast v2 — design fixed before results (2026-10-15)

User: robotics' own topics must be forecast too, and the forecast must consider more than "robotics trend + AI trend".
Author inflow dropped (user: "作者这个有那么重要吗？"; costly, overlaps signal 3).

## Topics
- R: the 58 robotics topics of ~/robotics-trends (rules checked on robotics text there; [term] items excluded).
- M: the 52 ML/CV topics in data/topics_ok.txt (rules checked on robotics text in step 2).
- Duplicates: an M topic is dropped when its Jaccard overlap (matched robotics papers 2020-2026) with some R topic is
  >= 0.6; the R version is kept. List written to results/v2_duplicates.csv.

## Target
Robotics change into year T of the logit of the standardised share (venue + year grouped binomial GLM, 2025 venue mix,
floor 0.05%). T = 2024, 2025 (full years); 2026 (partial: no IROS, CoRL) reported separately.

## Signals for topic X and target year T (all known at the end of T-1)
1. own   : OLS slope of the robotics logit over 2020..T-1. Expected sign +.
   level : robotics logit at T-1 (saturation). Expected sign −.
2. venue : early venues CoRL + RSS against the rest. g(t) = logit_lead(t) − logit_rest(t), raw pooled shares with
   (k + 0.5)/(n + 1). venue = g(T-1) − mean g(2020..T-2) (a topic unusually ahead at the early venues last year).
   In 2026 CoRL is missing, so the 2026 early-venue share is RSS only. Expected sign +.
3. emb   : ML/CV papers about robots (ML/CV rule "Robotics / embodied AI", 177 papers in 2020 to 1,111 in 2026). R topics
   use the robotics rules applied to their titles + abstracts; M topics their ML/CV matrix entries. Standardised share
   (venue + year GLM, 2024+2025 venue mix of the subset). emb = change T-2 -> T-1 of its logit. Expected sign +.
   Before use: the robotics rules are checked on these ML/CV papers (abstract-judged samples, as in step 2); topics
   below 0.8 precision get no emb signal.
4. diff  : method diffusion into the topic. Methods = R family "A learning" + all M topics, minus X and its
   duplicates. diff = sum over methods m of P(m | X, robotics papers of T-1) · (robotics logit change of m into T-1).
   Expected sign +.
5. mlcv  : M topics only. ML/CV change T-2 -> T-1 of the logit share, computed on the part of ML/CV papers whose
   company matches robotics: papers split into "names LLM / VLM / reasoning" and "does not"; the part whose co-topic
   log-lift profile in T-3..T-1 correlates more (Spearman) with the robotics profile of X in T-3..T-1 is used.
   Expected sign +.
6. new topics: separate analysis (not in the score). Word n-grams (1-3) in robotics + ML/CV-embodied abstracts that are
   not matched by any rule, ranked by growth from T-3..T-2 to T-1 (min 10 papers in T-1). Backtest: do the top 30 grow
   in robotics in T more than frequency-matched random n-grams? Then the 2026 list, read from abstracts before any is
   called a topic.

## Combination
- Each signal alone: Spearman with the target (no fitting), per T.
- Score: mean over the available signals of their z-scores across topics within T, with the expected signs. No fitted
  weights. Forecast change = a + b · score, a, b fitted on earlier target years (2023..T-1).
- Secondary: OLS on all signals with fitted weights (earlier target years), to see whether weights help.
- Baseline: own slope alone (best robotics-only model in step 8).
- Metrics: Spearman, MAE, top-10 overlap; R and M topics together and separately; bootstrap over topics of
  rho(score) − rho(own).
- Ablation: score without each signal.

## Forecast 2027
Score model fitted on target years 2023-2026; interval from the backtest residuals; for every topic the signals that
push it up or down are shown; signals that disagree are marked.

## Not modelled
Hardware cost, company and funding activity, release dates of datasets and models, IROS / CoRL 2026.

## Rule check of signal 3 (added before the review results)
Precision per topic and period = pool-weighted mix of title and abstract-only items, correct = Y + P (the rule found the
topic's sense; the shares count papers that name a topic, also in passing, as in the trend studies), "?" excluded.
Y-only precision reported alongside. emb signal used when precision >= 0.8 in 2024-2026 and in 2020-2023
(2020-2023 exempt when its pool is < 20); topics with < 20 matches in total get no emb signal.

## Amendment (2026-10-15, after building the series, before any backtest)
- 59 R topics (not 58), 42 M topics kept; 10 M topics duplicate an R topic (results/v2_duplicates.csv).
- Signal 5 (mlcv) also for an R topic that has a duplicate M topic: it uses that M topic's ML/CV series (highest Jaccard).
- For LLMs, Vision-language and Reasoning (the topics that define the split) signal 5 uses all ML/CV papers.

## Amendment after the backtest results (2026-10-15) — interval only, point forecasts unchanged
Out-of-sample errors depend on topic size (median |error| 0.29 for topics < 1% in the year before, 0.20 for 1-3%,
0.08 for >= 3%). The 2027 intervals therefore use the error quantiles of the topic's own size band (by 2026 share)
instead of one pooled set. Chosen after seeing the errors; it changes no forecast and no backtest score.
