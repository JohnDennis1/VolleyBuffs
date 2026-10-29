# Baseline models every candidate should be compared against.
# Run from the project root: Rscript baseline_models.R
# Writes predictions/<model>__<feature_set>.csv for each baseline.
#
#   base_rate__none        Training share of sets CU won; ignores the game state.
#   logistic__score        Score difference x total points, next server, fifth set.
#   logistic__score_skills Score inputs plus the 38 cumulative skill counts.
#   logistic__score_rates  Score inputs plus home-minus-away skill rates
#                          (see rate_features.R).
#
# A model that cannot beat logistic__score is not using the skill information
# usefully. These also serve as a template for adding a new model.

source("evaluation_functions.R")
source("rate_features.R")
live <- load_model_data()

# Weighted mean of y: with set weights, this is the share of sets CU won.
base_rate_fit <- function(x, y, w) weighted.mean(y == "win", w)
base_rate_predict <- function(model, x) rep(model, nrow(x))

# Quasibinomial accepts the fractional set weights. Coefficient p-values are
# not meaningful here (correlated rallies); only the predictions are used.
# Some counts are exact combinations of others (e.g. home + away serve
# attempts = rallies played). Those coefficients are inestimable, so drop
# them and refit rather than predict from a rank-deficient fit.
logistic_fit <- function(x, y, w) {
  train <- as.data.frame(x)
  train$win <- as.integer(y == "win")
  formula <- win ~ . + score_difference:total_points
  model <- glm(formula, data = train, weights = w, family = quasibinomial())
  aliased <- intersect(names(which(is.na(coef(model)))), colnames(x))
  if (length(aliased) > 0) {
    model <- glm(formula, data = train[setdiff(names(train), aliased)],
                 weights = w, family = quasibinomial())
  }
  model
}
logistic_predict <- function(model, x) {
  as.numeric(predict(model, newdata = as.data.frame(x), type = "response"))
}

run_model(live, live$x_score, base_rate_fit, base_rate_predict) |>
  save_predictions(model = "base_rate", feature_set = "none")
run_model(live, live$x_score, logistic_fit, logistic_predict) |>
  save_predictions(model = "logistic", feature_set = "score")
run_model(live, live$x_skills, logistic_fit, logistic_predict) |>
  save_predictions(model = "logistic", feature_set = "score_skills")
run_model(live, rate_features(live), impute_means(logistic_fit),
          impute_means_predict(logistic_predict)) |>
  save_predictions(model = "logistic", feature_set = "score_rates")
