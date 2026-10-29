"""Random forest (scikit-learn) on the three predictor sets.

Python counterpart of random_forest.qmd (ranger). Run from the repository
root in the stat environment:

    python random_forest.py
    Rscript compare_models.R

For each feature set (score, score_skills, score_rates, skills, rates):
  1. Tune leaf size and max_features with GridSearchCV over the season folds.
  2. Refit the best settings with more trees: out-of-fold predictions for the
     training seasons (cross_val_predict), then a fit on all training seasons
     to predict the test seasons.
  3. Save predictions/sklearn_rf__<feature_set>.csv and print importance.

Tuning results go to results/sklearn_rf_tuning.csv. Settings are chosen on CV
only; test predictions are saved for the final comparison, not examined here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict

from evaluation import KEYS, load_model_data, quick_brier, save_predictions

# Lets sample_weight reach both the forest's fit and the scorer (see below).
sklearn.set_config(enable_metadata_routing=True)

SEED = 4640
TUNING_TREES = 300
FINAL_TREES = 500
RESULTS_DIR = Path("results")

live = load_model_data()
train = (live.meta["role"] == "train").to_numpy()
test = (live.meta["role"] == "test").to_numpy()
y_train = live.y[train]
w_train = live.meta["set_weight"].to_numpy()[train]

# Season folds, not random ones. Rallies from the same set are near-copies
# with the same outcome, so a random split puts copies on both sides and
# scores look better than they are (by about 0.003 Brier here, as large as
# the differences between models we are trying to measure). Each training season is one fold
# (fold 1-4 -> 0-3); every rally of a match stays in its season's fold.
folds = PredefinedSplit(live.meta["fold"].to_numpy()[train].astype(int) - 1)
seasons = live.meta["season"].to_numpy()[train]
for fit_idx, val_idx in folds.split():
    assert not set(seasons[fit_idx]) & set(seasons[val_idx])

# serving_point_pct_diff equals sideout_pct_diff exactly (CU's serving-point
# % is 100 minus the opponent's sideout %), so it is dropped.
# Rates are NaN until both teams have attempted a skill. The forest handles
# NaN natively: each split learns which side missing values go to, so no
# imputation is needed.
#
# skills and rates drop the score inputs, to see what skills alone can do,
# e.g. at a tied score where the score inputs say little. rates also drops
# sideout_pct_diff: it is the share of points won while receiving, so it
# carries score information rather than skill. (Skill counts still partly add
# up to the score, since points come from kills, aces, and errors.)
score_rates = live.x_rates.drop(columns="serving_point_pct_diff")
feature_sets = {
    "score": live.x_score,
    "score_skills": live.x_skills,
    "score_rates": score_rates,
    "skills": live.x_skills.drop(columns=live.x_score.columns),
    "rates": score_rates.drop(columns=[*live.x_score.columns, "sideout_pct_diff"]),
}
max_features_grid = {
    "score": [1, 2, 3, None],
    "score_skills": [6, 14, 28, None],
    "score_rates": [3, 6, 11, None],
    "skills": [6, 14, 28, None],
    "rates": [3, 6, 9, None],
}
leaf_sizes = [10, 25, 50, 100, 200, 400, 800, 1600]

# Set weights make each set count equally, in fitting and in scoring. A
# weighted Brier score with weights 1 / (rallies in the set) is the average
# over sets of each set's mean Brier score, the same quantity compare_models.R
# reports. Out-of-bag error is not used: out-of-bag rallies have near-copies
# in the bag, so they are not truly unseen.
scoring = {
    "brier": get_scorer("neg_brier_score").set_score_request(sample_weight=True),
    "log_loss": get_scorer("neg_log_loss").set_score_request(sample_weight=True),
}
forest = RandomForestClassifier(n_estimators=TUNING_TREES, random_state=SEED)
forest.set_fit_request(sample_weight=True)

tuning, final_models = [], {}
for name, x in feature_sets.items():
    x_train = x[train]

    # 1. Tune. max_features=None means every predictor is tried at each split.
    search = GridSearchCV(
        forest,
        {"min_samples_leaf": leaf_sizes, "max_features": max_features_grid[name]},
        scoring=scoring, refit="brier", cv=folds, n_jobs=-1,
    )
    search.fit(x_train, y_train, sample_weight=w_train)
    results = pd.DataFrame(search.cv_results_)
    tuning.append(pd.DataFrame({
        "feature_set": name,
        "max_features": results["param_max_features"].fillna("all"),
        "min_samples_leaf": results["param_min_samples_leaf"],
        "brier": -results["mean_test_brier"],
        "log_loss": -results["mean_test_log_loss"],
    }))

    # 2. Refit the best settings with more trees.
    model = clone(search.best_estimator_).set_params(n_estimators=FINAL_TREES,
                                                     n_jobs=-1)
    cv_proba = cross_val_predict(model, x_train, y_train, cv=folds,
                                 params={"sample_weight": w_train},
                                 method="predict_proba")[:, 1]
    model.fit(x_train, y_train, sample_weight=w_train)
    final_models[name] = model

    # 3. Save in the shared format so compare_models.R scores it.
    pred = pd.concat([
        live.meta.loc[train, KEYS].assign(split="cv", win_probability=cv_proba),
        live.meta.loc[test, KEYS].assign(
            split="test", win_probability=model.predict_proba(x[test])[:, 1]),
    ])
    save_predictions(pred, model="sklearn_rf", feature_set=name)
    print(f"{name:13s} best {search.best_params_}  "
          f"CV Brier {quick_brier(live, pred[pred.split == 'cv']):.4f}")

tuning = pd.concat(tuning).sort_values(["feature_set", "brier"])
RESULTS_DIR.mkdir(exist_ok=True)
tuning.to_csv(RESULTS_DIR / "sklearn_rf_tuning.csv", index=False)

# GridSearchCV averages the four fold scores; compare_models.R averages over
# all sets. Folds hold 72-77 sets, so the two differ only slightly.
print("\nCV Brier by leaf size (best max_features at each size; mean of folds)")
print(tuning.groupby(["min_samples_leaf", "feature_set"])["brier"].min()
      .unstack().round(4).to_string())

# Impurity importance from the final fits: a description of what each forest
# splits on, not a test of predictive value.
print("\nImpurity importance (share of total, top 8)")
for name in ["score_skills", "score_rates"]:
    importance = pd.Series(final_models[name].feature_importances_,
                           index=feature_sets[name].columns)
    print(f"  {name}:")
    print(importance.sort_values(ascending=False).head(8).round(3).to_string())

print("\nRun `Rscript compare_models.R` for intervals, stages, and calibration.")
