# Rate feature set: score inputs plus home-minus-away skill rates.
# Source after evaluation_functions.R:
#   live <- load_model_data()
#   x_rates <- rate_features(live)
#   fit <- impute_means(fit_fun); predict <- impute_means_predict(predict_fun)
#
# Rates are percentages of attempts so far in the set (from
# Data_Wrangling_LivePred.R). A difference is NA until both teams have at
# least one attempt, so models must be wrapped with impute_means(), which
# fills NAs with means from the rows the model is fitted on. Under
# cv_predictions() that is the three fitting seasons only, so nothing from the
# predicted season leaks in.

# Freeball error % is left out: it is NA in 43% of training rallies.
# Attack efficiency is left out: it equals kill % - attack error % -
# blocked-attack %, so it adds no information to a model that has all three.
rate_diffs <- c("kill_pct", "attack_error_pct", "blocked_attack_pct",
                "ace_pct", "service_error_pct", "perfect_reception_pct",
                "reception_error_pct", "block_win_pct", "block_error_pct",
                "dig_error_pct", "set_error_pct", "sideout_pct",
                "serving_point_pct")

rate_features <- function(live) {
  d <- live$score_plus_skills
  x <- as.matrix(d[paste0(rate_diffs, "_raw_diff")])
  colnames(x) <- paste0(rate_diffs, "_diff")
  # Exposure: the same rate means more after 20 attacks than after 2.
  attack_attempts <- d$home_attack_attempts + d$away_attack_attempts
  cbind(live$x_score, x, attack_attempts = attack_attempts)
}

# Wrap a fit function so missing values are filled with set-weighted column
# means from the fitting rows. The means are stored with the model.
impute_means <- function(fit_fun) {
  function(x, y, w) {
    observed <- !is.na(x)
    means <- colSums(ifelse(observed, x, 0) * w) / colSums(observed * w)
    list(model = fit_fun(fill_missing(x, means), y, w), means = means)
  }
}
impute_means_predict <- function(predict_fun) {
  function(model, x) predict_fun(model$model, fill_missing(x, model$means))
}

fill_missing <- function(x, means) {
  for (j in which(colSums(is.na(x)) > 0)) {
    x[is.na(x[, j]), j] <- means[j]
  }
  x
}
