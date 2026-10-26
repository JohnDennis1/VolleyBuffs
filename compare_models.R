# Score every saved prediction file the same way.
# Run from the project root:
#   Rscript compare_models.R          cross-validation results (for development)
#   Rscript compare_models.R test     also report held-out test results
#
# Report the test results only for the final comparison. Choosing models or
# settings by their test scores makes the test estimate optimistic.
#
# Brier score and log loss: lower is better. Each set counts equally.
# 95% intervals resample whole matches. diff_vs_ref is the Brier difference
# from logistic__score (negative = better than the score-only baseline).

source("evaluation_functions.R")
show_test <- "test" %in% commandArgs(trailingOnly = TRUE)

live <- load_model_data()
pred <- read_predictions()
if (!show_test) pred <- filter(pred, split == "cv")

results <- score_predictions(pred, live$meta)

options(pillar.sigfig = 4, width = 120)
for (s in unique(results$summary$split)) {
  cat("\n==", if (s == "cv") "Cross-validation (training seasons)" else
        "Held-out test (2023 + 2024)", "==\n")
  print(filter(results$summary, split == s), n = Inf, width = Inf)
}
cat("\n== Brier score by stage of the set ==\n")
results$by_stage |>
  mutate(model = paste(model, feature_set, sep = "__")) |>
  select(-feature_set, -sets) |>
  arrange(stage) |>
  tidyr::pivot_wider(names_from = stage, values_from = brier) |>
  arrange(split, model) |>
  print(n = Inf, width = Inf)

write.csv(results$summary, file.path(prediction_dir, "comparison_summary.csv"),
          row.names = FALSE)
write.csv(results$by_stage, file.path(prediction_dir, "comparison_by_stage.csv"),
          row.names = FALSE)
write.csv(results$calibration, file.path(prediction_dir, "comparison_calibration.csv"),
          row.names = FALSE)
