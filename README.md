# Football Analytics — Match Result Prediction

Predicting English Premier League full-time results (Home / Draw / Away) from
pre-match form and season-to-date statistics, comparing 5 ML/DL models.

## Dataset

`data/raw/final_dataset.csv` — 6,840 EPL matches (2000/01–2017/18 seasons),
one row per fixture, with pre-match features engineered from the match
history available *before kickoff*:

- Season-to-date goals scored/conceded (`HTGS`/`ATGS`/`HTGC`/`ATGC`) and points (`HTP`/`ATP`)
- Last-5-match form strings (`HM1`–`HM5`, `AM1`–`AM5`) and form points
- Win/loss streaks (3- and 5-match), goal difference, `DiffPts`, `DiffFormPts`
- Match week (`MW`)

**Target:** the dataset ships an `FTR` column already binarized to `H`/`NH`
(home win vs. not). This project recomputes the true 3-class result
(`H`/`D`/`A`) directly from `FTHG`/`FTAG` (see [`src/football_ml/data.py`](src/football_ml/data.py))
so draws are modelled explicitly rather than folded into "not a home win."

`FTHG`, `FTAG`, and the original `FTR` are dropped before training — they
encode the outcome directly and would leak the label.

## Models

All 5 share one preprocessing pipeline (`StandardScaler` on numeric
features, `OneHotEncoder` on `HomeTeam`/`AwayTeam`/form columns — see
[`src/football_ml/features.py`](src/football_ml/features.py)), so
differences in results reflect the estimator, not the inputs:

| Model | Implementation |
|---|---|
| Logistic Regression | `sklearn`, `class_weight="balanced"` |
| Random Forest | `sklearn`, 400 trees |
| LightGBM | gradient-boosted trees |
| SVM | RBF kernel |
| MLP (deep learning) | PyTorch feed-forward net (128→64), early-stopped on a chronological validation split |

## Evaluation

Matches are sorted by date; the most recent 20% (1,368 matches, roughly the
final 2 seasons) are held out as a **chronological test set** — no future
match is ever used to inform a prediction about an earlier one. The training
slice is further validated with `TimeSeriesSplit` (5-fold, expanding window).

## Results (chronological holdout test set)

| Model | CV accuracy | Test accuracy | Test macro F1 | Test log loss |
|---|---|---|---|---|
| Logistic Regression | 0.473 | 0.482 | 0.462 | 1.037 |
| Random Forest | 0.516 | 0.512 | 0.435 | 0.999 |
| LightGBM | 0.483 | 0.485 | 0.451 | 1.029 |
| SVM | 0.471 | 0.468 | 0.447 | 0.991 |
| MLP (PyTorch) | 0.494 | 0.485 | 0.446 | 1.006 |

Random Forest edges out the others on accuracy; Logistic Regression has the
best macro F1 (it predicts draws more often, at some cost to overall
accuracy). All models land in the high-0.40s/low-0.50s range — consistent
with how hard EPL outcomes are to call from pre-match stats alone: a
"predict the home team always wins" baseline already scores ~46% given the
class balance in this data, and full match-outcome prediction is well known
to be a low-signal problem even for bookmakers' own models.

Per-model confusion matrices, the model comparison chart, and SHAP feature
importance (Random Forest / LightGBM) are in [`results/`](results/).
The top SHAP drivers are `DiffPts`, `DiffFormPts`, goal-difference and
points-so-far features — teams that are already ahead on points/form before
kickoff are (unsurprisingly) the strongest predictor of the result.

## Project structure

```
data/raw/final_dataset.csv      raw data
src/football_ml/
  config.py                     paths, feature lists, palette
  data.py                       loading, target recomputation, chronological split
  features.py                   shared preprocessing pipeline
  models.py                     the 5 model pipelines
  mlp.py                        PyTorch MLP, wrapped sklearn-style
  evaluate.py                   TimeSeriesSplit CV + holdout metrics
  shap_utils.py                 SHAP explainability for tree models
  plotting.py                   chart styling
scripts/train_all.py            trains + evaluates all 5 models, writes results/
results/                        metrics, comparison table, confusion matrices, SHAP plots
models/                         fitted model artifacts (gitignored — regenerate via the script)
```

## Running it

```bash
pip install -r requirements.txt
python scripts/train_all.py
```
