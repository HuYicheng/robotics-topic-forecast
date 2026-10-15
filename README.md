# Robotics Topic Forecast

**Page:** https://huyicheng.github.io/robotics-topic-forecast/

Which topics will grow or shrink in robotics papers (ICRA, IROS, RSS, CoRL, T-RO, IJRR, Science Robotics) in 2027? The
forecast covers 59 robotics topics (from [Robotics Topic Trends](https://huyicheng.github.io/robotics-topic-trends/)) and
42 ML / vision topics as robotics uses them (from [ML & CV Topic Trends](https://huyicheng.github.io/ml-cv-topic-trends/)),
and combines six signals, each backtested on 2024-2026. The design was written down before the backtest: `DESIGN_v2.md`.

## Signals (known at the end of year T-1, for the change into year T)

| signal | what it measures | rank correlation with the actual change, 2024 / 2025 / 2026 (partial) |
|---|---|---|
| own slope | the topic's robotics trend since 2020 | 0.26 / 0.40 / 0.46 |
| room to grow | common topics grow more slowly | 0.22 / 0.37 / 0.10 |
| early venues | the topic ahead at CoRL + RSS compared with usual | 0.17 / 0.05 / 0.15 |
| ML/CV robot papers | change among robot papers at NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV (rules checked on those papers) | 0.43 / 0.65 / 0.30 |
| methods spreading in | rising methods already used in the topic's papers | 0.48 / 0.56 / 0.51 |
| ML/CV overall | change at ML / vision venues, LLM-related and other papers separated | 0.20 / 0.43 / 0.48 |
| **combined score** | equal-weight mean of the scaled signals | **0.47 / 0.66 / 0.55** |

Robotics' own slope alone: 0.26 / 0.40 / 0.46 (bootstrap advantage of the score: +0.20 [0.04, 0.37], +0.25 [0.08, 0.42],
+0.08 [-0.08, 0.24]). Fitted weights did worse than equal weights. Fast-growing new phrases were also tested and did not
predict anything (`results/v2_emerging_backtest.csv`). Intervals come from the backtest errors of topics of the same
size. Because almost every topic is named a little more often each year, the page judges direction against the average
topic.

## Limits

Three target years, one partial; robotics 2026 lacks IROS and CoRL and ML/CV 2026 lacks NeurIPS; keyword rules count
papers that name a topic; hardware, funding and model or dataset releases are not in the data.

## Files

| step | script | output |
|---|---|---|
| 1-7 | first version (ML/CV lead only) | data/series.csv, results/backtest.csv, forecast_2027.csv, ... |
| 8-9 | own-trend and topic-alignment checks | results/own_trend*.csv, results/alignment.csv |
| 10, 14 | ML/CV robot papers, rule check | data/embodied_matrix.csv, results/emb_precision.csv |
| 11 | series for all signals | data/v2_*.csv |
| 12 | signals, backtest, 2027 forecast | results/v2_*.csv |
| 13 | new phrases | results/v2_emerging_*.csv |
| 15 | page | data.json, index.html (template site_src/index_v2.html) |

The scripts expect the two companion projects next to them. Abstract texts and the reading samples are not included.
