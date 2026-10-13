# Shared evaluation tools: every model uses the same data, split, prediction
# format, and scoring. Source this file from a model script:
#   source("evaluation_functions.R")
# Requires dplyr. Run Data_Wrangling_LivePred.R first; the split comes from
# the committed model_splits.csv (see make_splits.R).
#
# Workflow for a new model:
#   live <- load_model_data()
#   fit_fun <- function(x, y, w) { ...return a fitted model... }
#   predict_fun <- function(model, x) { ...return P(CU wins set)... }
#   pred <- run_model(live, live$x_skills, fit_fun, predict_fun)
#   save_predictions(pred, model = "my_model", feature_set = "score_skills")
# Then run compare_models.R.
#
# x is a numeric matrix (rows = snapshots), y is a factor with levels
# loss/win, and w holds set weights (each set sums to 1). Models that cannot
# use weights may ignore w.

library(dplyr)

prediction_dir <- "predictions"

# Load the wrangled data and attach each snapshot's season, role, and fold.
load_model_data <- function(live_file = "Data/derived/live_analysis.rds",
                            splits_file = "model_splits.csv") {
  live <- readRDS(live_file)
  splits <- read.csv(splits_file)
  row_split <- match(live$score_plus_skills$match_id, splits$match_id)
  stopifnot(!anyNA(row_split))
  live$meta <- live$score_plus_skills |>
    select(match_id, set_number, point_id, rally_number, total_points,
           fifth_set, home_set_win, set_weight) |>
    mutate(season = splits$season[row_split],
           role = splits$role[row_split],
           fold = splits$fold[row_split])
  live
}

# Out-of-fold predictions for training matches: each season is predicted by
# a model fitted on the other training seasons. Use these for tuning.
cv_predictions <- function(live, x, fit_fun, predict_fun) {
  meta <- live$meta
  folds <- sort(unique(meta$fold[meta$role == "train"]))
  probability <- rep(NA_real_, nrow(meta))
  for (f in folds) {
    fit_rows <- meta$role == "train" & meta$fold != f
    predict_rows <- meta$role == "train" & meta$fold == f
    model <- fit_fun(x[fit_rows, , drop = FALSE], live$y[fit_rows],
                     meta$set_weight[fit_rows])
    probability[predict_rows] <- predict_fun(model, x[predict_rows, , drop = FALSE])
  }
  keep <- meta$role == "train"
  tibble(select(meta[keep, ], match_id, set_number, point_id),
         split = "cv", win_probability = probability[keep])
}

# Final predictions: fit on all training matches, predict the test matches.
# Run this once settings are chosen; do not tune against its results.
test_predictions <- function(live, x, fit_fun, predict_fun) {
  meta <- live$meta
  fit_rows <- meta$role == "train"
  test_rows <- meta$role == "test"
  model <- fit_fun(x[fit_rows, , drop = FALSE], live$y[fit_rows],
                   meta$set_weight[fit_rows])
  tibble(select(meta[test_rows, ], match_id, set_number, point_id),
         split = "test",
         win_probability = predict_fun(model, x[test_rows, , drop = FALSE]))
}

run_model <- function(live, x, fit_fun, predict_fun) {
  bind_rows(cv_predictions(live, x, fit_fun, predict_fun),
            test_predictions(live, x, fit_fun, predict_fun))
}

# Standard prediction file: predictions/<model>__<feature_set>.csv
save_predictions <- function(pred, model, feature_set) {
  stopifnot(all(c("match_id", "set_number", "point_id", "split",
                  "win_probability") %in% names(pred)),
            all(is.finite(pred$win_probability)),
            all(pred$win_probability >= 0 & pred$win_probability <= 1),
            !grepl("__", model), !grepl("__", feature_set))
  dir.create(prediction_dir, showWarnings = FALSE)
  pred <- mutate(pred, model = model, feature_set = feature_set, .before = 1)
  path <- file.path(prediction_dir, paste0(model, "__", feature_set, ".csv"))
  write.csv(pred, path, row.names = FALSE)
  invisible(path)
}

read_predictions <- function() {
  files <- list.files(prediction_dir, pattern = "__.*\\.csv$", full.names = TRUE)
  stopifnot(length(files) > 0)
  bind_rows(lapply(files, read.csv))
}

# Brier score and log loss for each snapshot. Probabilities are clipped for
# log loss so a confident 0 or 1 gives a large but finite penalty.
add_losses <- function(pred, meta) {
  pred |>
    inner_join(meta, by = c("match_id", "set_number", "point_id")) |>
    mutate(
      brier = (win_probability - home_set_win)^2,
      clipped = pmin(pmax(win_probability, 1e-8), 1 - 1e-8),
      log_loss = -(home_set_win * log(clipped) +
                     (1 - home_set_win) * log(1 - clipped)),
      stage = factor(case_when(fifth_set == 1 ~ "Fifth set",
                               total_points <= 15 ~ "Early (1-15 points)",
                               total_points <= 30 ~ "Middle (16-30 points)",
                               TRUE ~ "Late (31+ points)"),
                     levels = c("Early (1-15 points)", "Middle (16-30 points)",
                                "Late (31+ points)", "Fifth set"))
    )
}

# Each set counts equally: average within a set, then across sets.
# Confidence intervals resample whole matches (rallies and sets within a
# match are correlated). Differences are paired against `reference`.
score_predictions <- function(pred, meta, reference = "logistic__score",
                              reps = 2000, seed = 4640) {
  losses <- add_losses(pred, meta) |>
    mutate(model_id = paste(model, feature_set, sep = "__"))

  per_set <- losses |>
    group_by(model_id, model, feature_set, split, match_id, set_number) |>
    summarise(brier = mean(brier), log_loss = mean(log_loss), .groups = "drop")

  summary <- per_set |>
    group_by(model, feature_set, split) |>
    summarise(sets = n(), brier = mean(brier), log_loss = mean(log_loss),
              .groups = "drop")

  # Bootstrap by match, separately for each split.
  set.seed(seed)
  intervals <- list()
  for (s in unique(per_set$split)) {
    by_match <- per_set |>
      filter(split == s) |>
      group_by(model_id, match_id) |>
      summarise(brier_sum = sum(brier), sets = n(), .groups = "drop")
    sums <- tidyr::pivot_wider(by_match, id_cols = match_id, names_from = model_id,
                               values_from = brier_sum)
    n_sets <- by_match |> distinct(match_id, sets)
    n_sets <- n_sets$sets[match(sums$match_id, n_sets$match_id)]
    sums <- as.matrix(sums[, -1])
    counts <- matrix(0L, reps, nrow(sums))
    for (b in seq_len(reps)) {
      counts[b, ] <- tabulate(sample.int(nrow(sums), replace = TRUE), nrow(sums))
    }
    boot <- (counts %*% sums) / as.vector(counts %*% n_sets)
    has_ref <- reference %in% colnames(boot)
    intervals[[s]] <- tibble(
      model_id = colnames(boot), split = s,
      brier_low = apply(boot, 2, quantile, 0.025),
      brier_high = apply(boot, 2, quantile, 0.975),
      diff_vs_ref = if (has_ref) colMeans(boot - boot[, reference]) else NA_real_,
      diff_low = if (has_ref) apply(boot - boot[, reference], 2, quantile, 0.025) else NA_real_,
      diff_high = if (has_ref) apply(boot - boot[, reference], 2, quantile, 0.975) else NA_real_
    )
  }
  summary <- summary |>
    mutate(model_id = paste(model, feature_set, sep = "__")) |>
    left_join(bind_rows(intervals), by = c("model_id", "split")) |>
    select(-model_id) |>
    arrange(split, brier)

  by_stage <- losses |>
    group_by(model, feature_set, split, stage, match_id, set_number) |>
    summarise(brier = mean(brier), .groups = "drop") |>
    group_by(model, feature_set, split, stage) |>
    summarise(sets = n(), brier = mean(brier), .groups = "drop")

  calibration <- losses |>
    mutate(bin = pmin(floor(win_probability * 10), 9L) / 10) |>
    group_by(model, feature_set, split, bin) |>
    summarise(predicted = weighted.mean(win_probability, set_weight),
              observed = weighted.mean(home_set_win, set_weight),
              snapshots = n(), .groups = "drop")

  list(summary = summary, by_stage = by_stage, calibration = calibration)
}
