"""Python version of the shared model tools in evaluation_functions.R.

Same data, split, feature sets, and prediction format as the R models, so a
Python model's predictions are scored by compare_models.R alongside them.
Scoring stays in R; this file only loads data, runs the folds, and saves.

Run Data_Wrangling_LivePred.R first (it writes
Data/derived/live_score_plus_skills.csv). Run scripts from the repository root.

Workflow for a new model:

    from evaluation import load_model_data, run_model, save_predictions

    live = load_model_data()

    def fit_fun(x, y, w):        # x: DataFrame, y: 0/1 array, w: set weights
        ...return a fitted model...

    def predict_fun(model, x):   # return P(CU wins the set) as a 1-d array
        ...

    pred = run_model(live, live.x_rates, fit_fun, predict_fun)
    save_predictions(pred, model="my_model", feature_set="score_rates")

Then run `Rscript compare_models.R`. While tuning, use cv_predictions() alone;
run_model() also fits on all training seasons and predicts the test seasons.

Unlike the R version, y is 0/1 (1 = CU wins the set) rather than a loss/win
factor. Set weights make each set sum to 1; models that cannot use weights may
ignore w.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_DIR = Path("predictions")
KEYS = ["match_id", "set_number", "point_id"]

# Column lists match Data_Wrangling_LivePred.R and rate_features.R, in order.
SCORE_PREDICTORS = ["score_difference", "total_points", "next_home_serves",
                    "fifth_set"]
BASE_COUNTS = ["attack_attempts", "kills", "attack_errors", "blocked_attacks",
               "serve_attempts", "aces", "serve_errors", "reception_attempts",
               "perfect_receptions", "reception_errors", "block_attempts",
               "winning_blocks", "block_errors", "dig_attempts", "dig_errors",
               "set_attempts", "set_errors", "freeball_attempts",
               "freeball_errors"]
SKILL_PREDICTORS = SCORE_PREDICTORS + [f"{side}_{count}" for count in BASE_COUNTS
                                       for side in ("home", "away")]
RATE_DIFFS = ["kill_pct", "attack_error_pct", "blocked_attack_pct", "ace_pct",
              "service_error_pct", "perfect_reception_pct",
              "reception_error_pct", "block_win_pct", "block_error_pct",
              "dig_error_pct", "set_error_pct", "sideout_pct",
              "serving_point_pct"]


@dataclass
class LiveData:
    """Rows are rally snapshots; every table shares the same row order."""
    data: pd.DataFrame       # every wrangled column (counts, rates, ratios, ...)
    meta: pd.DataFrame       # keys, outcome, weight, season, role, fold
    y: np.ndarray            # 1 = CU wins the set
    x_score: pd.DataFrame    # 4 score inputs
    x_skills: pd.DataFrame   # score inputs + 38 cumulative skill counts
    x_rates: pd.DataFrame    # score inputs + 13 rate differences + attack attempts


def load_model_data(live_file="Data/derived/live_score_plus_skills.csv",
                    splits_file="model_splits.csv"):
    """Load the wrangled snapshots and attach each one's season, role, and fold."""
    data = pd.read_csv(live_file)
    splits = pd.read_csv(splits_file, dtype={"season": str})
    split_cols = splits.set_index("match_id")[["season", "role", "fold"]]
    assert data["match_id"].isin(split_cols.index).all(), "match missing from split"

    meta = data[KEYS + ["rally_number", "total_points", "fifth_set",
                        "home_set_win", "set_weight"]].join(
        split_cols, on="match_id")

    rates = data[[f"{r}_raw_diff" for r in RATE_DIFFS]]
    rates.columns = [f"{r}_diff" for r in RATE_DIFFS]
    x_rates = pd.concat([data[SCORE_PREDICTORS], rates], axis=1)
    # Exposure: the same rate means more after 20 attacks than after 2.
    x_rates["attack_attempts"] = data["home_attack_attempts"] + data["away_attack_attempts"]

    x_score = data[SCORE_PREDICTORS]
    x_skills = data[SKILL_PREDICTORS]
    assert x_score.notna().all().all() and x_skills.notna().all().all()
    return LiveData(data=data, meta=meta, y=data["home_set_win"].to_numpy(),
                    x_score=x_score, x_skills=x_skills, x_rates=x_rates)


def cv_predictions(live, x, fit_fun, predict_fun):
    """Out-of-fold predictions for training matches: each season is predicted
    by a model fitted on the other training seasons. Use these for tuning."""
    meta = live.meta
    train = (meta["role"] == "train").to_numpy()
    fold = meta["fold"].to_numpy()
    probability = np.full(len(meta), np.nan)
    for f in sorted(np.unique(fold[train])):
        fit_rows = train & (fold != f)
        predict_rows = train & (fold == f)
        model = fit_fun(x[fit_rows], live.y[fit_rows],
                        meta["set_weight"].to_numpy()[fit_rows])
        probability[predict_rows] = predict_fun(model, x[predict_rows])
    out = meta.loc[train, KEYS].copy()
    out["split"] = "cv"
    out["win_probability"] = probability[train]
    return out


def test_predictions(live, x, fit_fun, predict_fun):
    """Fit on all training matches, predict the test matches. Run this once
    settings are chosen; do not tune against its results."""
    meta = live.meta
    fit_rows = (meta["role"] == "train").to_numpy()
    test_rows = (meta["role"] == "test").to_numpy()
    model = fit_fun(x[fit_rows], live.y[fit_rows],
                    meta["set_weight"].to_numpy()[fit_rows])
    out = meta.loc[test_rows, KEYS].copy()
    out["split"] = "test"
    out["win_probability"] = predict_fun(model, x[test_rows])
    return out


def run_model(live, x, fit_fun, predict_fun):
    return pd.concat([cv_predictions(live, x, fit_fun, predict_fun),
                      test_predictions(live, x, fit_fun, predict_fun)])


def save_predictions(pred, model, feature_set):
    """Write the standard file predictions/<model>__<feature_set>.csv."""
    p = pred["win_probability"].to_numpy()
    assert set(KEYS + ["split", "win_probability"]) <= set(pred.columns)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert "__" not in model and "__" not in feature_set
    PREDICTION_DIR.mkdir(exist_ok=True)
    out = pred[KEYS + ["split", "win_probability"]].copy()
    out.insert(0, "feature_set", feature_set)
    out.insert(0, "model", model)
    path = PREDICTION_DIR / f"{model}__{feature_set}.csv"
    out.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return path


def impute_means(fit_fun):
    """Wrap a fit function so missing values (rates before a team's first
    attempt) are filled with set-weighted means of the fitting rows. Under
    cv_predictions() that is the fitting seasons only, so nothing leaks."""
    def fit(x, y, w):
        observed = x.notna()
        means = (x.fillna(0).mul(w, axis=0).sum()
                 / observed.mul(w, axis=0).sum())
        return {"model": fit_fun(x.fillna(means), y, w), "means": means}
    return fit


def impute_means_predict(predict_fun):
    def predict(model, x):
        return predict_fun(model["model"], x.fillna(model["means"]))
    return predict


def quick_brier(live, pred):
    """Set-averaged Brier score, the same headline number as compare_models.R
    (without intervals). Handy for tuning loops."""
    scored = pred.merge(live.meta[KEYS + ["home_set_win"]], on=KEYS)
    scored["brier"] = (scored["win_probability"] - scored["home_set_win"]) ** 2
    return scored.groupby(["match_id", "set_number"])["brier"].mean().mean()


# Usage example: python evaluation.py
# Runs two models through season cross-validation and prints a quick CV Brier
# score. It saves nothing; for real results, save_predictions() and then
# compare_models.R (which adds intervals, stages, and calibration).
if __name__ == "__main__":
    live = load_model_data()

    # 1. Base rate: always predict the weighted share of training sets CU won.
    def base_rate_fit(x, y, w):
        return np.average(y, weights=w)

    def base_rate_predict(model, x):
        return np.full(len(x), model)

    pred = cv_predictions(live, live.x_score, base_rate_fit, base_rate_predict)
    print(f"base_rate  score        CV Brier {quick_brier(live, pred):.4f}")  # R: 0.2521

    # 2. Logistic regression on the rate features. Rates are missing before a
    # team's first attempt, so the model is wrapped with impute_means().
    # C=np.inf turns off scikit-learn's default L2 penalty, like R's glm().
    # R's logistic__score_rates also has a score_difference x total_points
    # interaction, so this simpler version scores slightly differently.
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("Install scikit-learn for the logistic example: "
              "conda install -n stat -c conda-forge scikit-learn")
    else:
        def logistic_fit(x, y, w):
            return LogisticRegression(C=np.inf, max_iter=5000).fit(
                x, y, sample_weight=w)

        def logistic_predict(model, x):
            return model.predict_proba(x)[:, 1]

        pred = cv_predictions(live, live.x_rates, impute_means(logistic_fit),
                              impute_means_predict(logistic_predict))
        print(f"logistic   score_rates  CV Brier {quick_brier(live, pred):.4f}")
        # To add it to the comparison table:
        # save_predictions(run_model(live, live.x_rates, impute_means(logistic_fit),
        #                            impute_means_predict(logistic_predict)),
        #                  model="sklearn_logistic", feature_set="score_rates")
