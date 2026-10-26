# VolleyBuffs

A repository for the Senior Data Science Capstone (STAT 4640).

**Question:** At any point during a set, how likely is the University of Colorado
(CU) women's volleyball team to win that set — and do in-set skill statistics
(kills, aces, errors, passing quality) improve that prediction beyond the score
alone?

After every completed rally, a model sees the current score, who serves next,
and each team's running skill totals for the set, and outputs the probability
that CU wins the set.

## Data

Play-by-play scouting data for **104 CU conference matches** (168,769 event
records), scouted from CU's perspective.

| Season | Conference | Matches | Sets |
|---|---|---|---|
| Fall 2018 | Pac-12 | 1 | 5 |
| Fall 2019 | Pac-12 | 20 | 72 |
| Spring 2021 (COVID-delayed 2020 season) | Pac-12 | 20 | 75 |
| Fall 2021 | Pac-12 | 20 | 72 |
| Fall 2022 | Pac-12 | 20 | 74 |
| Fall 2023 | Pac-12 | 20 | 74 |
| Fall 2024 (first three matches) | Big 12 | 3 | 13 |

Notes:
- CU is recorded as the **home team in every match**, regardless of venue, so
  "home" means CU and "away" means CU's opponent.
- Opponent `UT` is Utah, not Texas.
- **The data is not in this repository and must not be committed.** Place the
  export at `Data/conference_matches_raw.csv` (the `Data/` folder is ignored by
  git). `make_splits.R` also reads `Data/conference_matches.csv` for match
  dates, but only if you regenerate the split.

## Pipeline

```
Data/conference_matches_raw.csv           one row per touch (~169k rows)
        │  Data_Wrangling_LivePred.R
        ▼
Data/derived/live_analysis.rds            one row per completed rally (16,610 snapshots, 385 sets)
        │  + model_splits.csv             shared season-holdout split (committed)
        ▼
model scripts  ──►  predictions/<model>__<feature_set>.csv
        │  compare_models.R
        ▼
one comparison table: Brier score, log loss, intervals, stage, calibration
```

### 1. Wrangling — `Data_Wrangling_LivePred.R`

Turns the event log into **rally snapshots**: one row after each completed
rally, describing the set at that moment.

- Excludes three sets with invalid scoring or serving records.
- Encodes every skill action (serve, reception, set, attack, block, dig,
  freeball) as attempts, successes, and errors from its evaluation code, then
  accumulates them within each set for both teams.
- Counts only actions recorded at or before the end of the rally, so no
  snapshot uses future information.
- Drops each set's final rally (the outcome is already known) and weights each
  remaining snapshot by `1 / snapshots in its set`, so every set counts equally.
- Saves `Data/derived/live_analysis.rds`, which contains the modeling tables, the
  predictor matrices `x_score` (4 score inputs) and `x_skills` (4 score inputs +
  38 skill counts), and the response `y` (CU wins the set: `loss` / `win`).

### 2. Split — `make_splits.R` → `model_splits.csv`

A **season holdout**, so models are judged on seasons they have never seen.

| Role | Fold | Seasons | Matches | Sets |
|---|---|---|---|---|
| Test (used once, for the final comparison) | — | 2023 + 2024 | 23 | 87 |
| Train | 1 | 2018 + 2019 | 21 | 77 |
| Train | 2 | Spring 2021 | 20 | 75 |
| Train | 3 | Fall 2021 | 20 | 72 |
| Train | 4 | 2022 | 20 | 74 |

Tuning uses 4-fold cross-validation with one season per fold: fit on three
training seasons, predict the fourth. All snapshots from a match stay together.
The split is committed as a file so every model uses exactly the same matches.

### 3. Evaluation — `evaluation_functions.R`, `compare_models.R`

Every model is scored identically:

- **Brier score** and **log loss** (lower is better), averaged within each set
  and then across sets.
- **95% intervals** from resampling whole matches, plus the paired difference
  from the score-only logistic baseline.
- Brier score by **stage of the set** (early / middle / late / fifth set) and a
  **calibration** table.

`compare_models.R` reports cross-validation results by default. Run it with
`test` only for the final comparison — choosing models by their test scores
makes the test estimate optimistic.

## Current results

Cross-validation over the four training seasons (298 sets):

| Model | Brier | Log loss | Brier vs score-only (95% interval) |
|---|---|---|---|
| Score-only logistic | **0.158** | **0.472** | — |
| Score + skills logistic | 0.172 | 0.521 | +0.014 (+0.004, +0.025) |
| Base rate (always 43%) | 0.252 | 0.698 | +0.094 (+0.077, +0.112) |

Adding all 38 raw skill counts to a logistic model makes predictions on unseen
seasons *worse* than the score alone. The score-only model is the bar for every
candidate model. Test-set results have not yet been examined.

## Running it

Requires R (4.6 tested) with `dplyr` and `tidyr`; individual models need their
own packages (e.g. `ranger`, `e1071`).

```sh
Rscript Data_Wrangling_LivePred.R   # build Data/derived/live_analysis.rds
Rscript baseline_models.R           # refit baselines (predictions are already committed)
Rscript compare_models.R            # cross-validation comparison
Rscript compare_models.R test       # final comparison only
```

Run all scripts from the repository root.

## Adding a model

Write a fit function and a predict function, then save predictions in the
standard format. `baseline_models.R` is a complete example.

```r
source("evaluation_functions.R")
live <- load_model_data()

fit_fun     <- function(x, y, w) ranger::ranger(x = x, y = y, case.weights = w,
                                                probability = TRUE)
predict_fun <- function(model, x) predict(model, x)$predictions[, "win"]

run_model(live, live$x_skills, fit_fun, predict_fun) |>
  save_predictions(model = "random_forest", feature_set = "score_skills")
```

- `x`: numeric predictor matrix; `y`: factor (`loss` / `win`); `w`: set weights
  (models that cannot use weights may ignore them).
- `predict_fun` must return the probability that CU wins the set.
- Use `cv_predictions()` alone while tuning; `run_model()` also fits the final
  model and predicts the test seasons.
- Then run `compare_models.R`.

## Repository contents

| File | Purpose |
|---|---|
| `Data_Wrangling_LivePred.R` | **Current** wrangling pipeline for live prediction |
| `make_splits.R`, `model_splits.csv` | Shared season-holdout split and CV folds |
| `evaluation_functions.R` | Shared loading, cross-validation, prediction, and scoring tools |
| `baseline_models.R` | Base-rate and logistic baselines |
| `compare_models.R` | Scores all saved predictions |
| `predictions/` | Saved model predictions (comparison tables are generated, not committed) |
| `wrangle_set_skills.R` | Whole-set (one row per team per set) summaries; not used for live prediction |
| `Volleybuffs_EDA.qmd` | Initial exploration of the data structure |
| `live_set_models.R` | Earlier live-prediction pipeline and logistic models; superseded by the files above |
| `CapstoneCodeRatios.1.Rmd` | Earliest exploratory whole-set models |
| `SVM_Kernel.qmd` | SVM model (in progress) |

Other models are developed on their own branches (`naive-bayes`,
`random-forest`, `SVM_Kernel`) and compared through `predictions/`.
