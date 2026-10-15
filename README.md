# Robotics Topic Forecast

**Page:** https://huyicheng.github.io/robotics-topic-forecast/

Which machine-learning and vision topics will grow in robotics papers in 2027? ML & CV venues tend to move first. This
forecasts next year's change at ICRA, IROS, RSS, CoRL, T-RO, IJRR and Science Robotics from last year's change at
NeurIPS, ICML, ICLR, CVPR, ICCV and ECCV, and backtests how well that worked for 2024-2026. Companion to
[Robotics Topic Trends](https://huyicheng.github.io/robotics-topic-trends/) and
[ML & CV Topic Trends](https://huyicheng.github.io/ml-cv-topic-trends/).

## What is measured

- **Papers**: 22,629 robotics research articles with abstracts, 2020-2026 (2026 lacks IROS and CoRL), from Robotics
  Topic Trends; 85,364 ML / CV main-track papers, 2020-2026 (2026 lacks NeurIPS), from ML & CV Topic Trends.
- **Topics**: the ML & CV topic rules, unchanged, matched on robotics titles and abstracts. Each rule was checked on
  robotics papers: random matches from 2020-2023 and 2024-2026 were read and judged from their abstracts. 52 topics with
  precision >= 0.8 in both periods are forecast (`data/topics_ok.txt`, `results/robotics_precision.csv`).
- **Shares**: standardised to a fixed venue mix (robotics 2025; ML / CV 2024 + 2025) with a logistic model of topic on
  venue and year (`data/series.csv`).
- **Model**: on the log-odds of the share, robotics change into year T = a + b × ML / CV change into year T-1, pooled
  over topics. Chosen by a backtest designed before its results were seen (`scripts/04_backtest.py`): rank correlation
  with the actual change 0.58 / 0.64 / 0.64 for 2024 / 2025 / 2026 against 0.10 / 0.30 / 0.51 for robotics' own trend;
  bootstrap over topics in `results/bootstrap.csv`. Intervals: percentiles of the backtest errors.
- **Limits**: few years per topic; both 2026 values are partial years filled in by the venue model; the same-year ML / CV
  change predicts about as well, so the data cannot say whether robotics follows ML / CV or both follow the same wave;
  the ML / CV rise of reinforcement learning in 2026 comes from RL used to train language models.

## Files

| step | script | output |
|---|---|---|
| 1 | scripts/01_apply_rules.py | data/robotics_mlcv_matrix.csv (paper × topic) |
| 2 | scripts/02_review_samples.py | samples read for the rule check (not included: they hold abstracts) |
| 3 | scripts/03_series.py | data/series.csv |
| 4 | scripts/04_backtest.py | results/backtest.csv |
| 5 | scripts/05_forecast.py | results/forecast_2027.csv, results/forecast_fit.txt |
| 6 | scripts/06_bootstrap.py | results/bootstrap.csv |
| 7 | scripts/07_site_data.py + site_src/index.html | data.json, index.html (the page) |

The scripts expect the two companion projects next to them. Abstract texts are not redistributed here.
