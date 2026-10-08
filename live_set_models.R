# Source after conference_data is created in Volleybuffs_EDA.qmd.
# Predictions are made AFTER a completed rally, before the next rally.
library(dplyr)
library(tidyr)

keys <- c("match_id", "set_number", "point_id")
point_records <- conference_data |>
  filter(point %in% TRUE) |>
  arrange(match_id, set_number, file_line_number)
stopifnot(!anyDuplicated(point_records[keys]))

# Audit complete scoring sequences before using a set for model development.
set_audit <- point_records |>
  group_by(match_id, set_number) |>
  summarise(
    target = if_else(first(set_number) == 5, 15L, 25L),
    complete = last(pmax(home_team_score, visiting_team_score)) >= target &
      abs(last(home_team_score - visiting_team_score)) >= 2,
    scores_valid = all(!is.na(home_team_score) & !is.na(visiting_team_score) &
      home_team_score >= 0 & visiting_team_score >= 0) &
      all(diff(c(0, home_team_score)) %in% 0:1) &
      all(diff(c(0, visiting_team_score)) %in% 0:1) &
      all(home_team_score + visiting_team_score == row_number()),
    winners_valid = all(coalesce(point_won_by == if_else(
      home_team_score > lag(home_team_score, default = 0L),
      home_team, visiting_team), FALSE)),
    no_early_finish = !any(head(
      pmax(home_team_score, visiting_team_score) >= target &
        abs(home_team_score - visiting_team_score) >= 2, -1)),
    home_set_win = as.integer(last(home_team_score) > last(visiting_team_score)),
    .groups = "drop"
  ) |>
  mutate(usable = coalesce(complete & scores_valid & winners_valid &
                            no_early_finish, FALSE))

# A kill is defined by the Attack/# combination. Retain winning_attack in
# the source and report disagreements rather than silently mixing definitions.
kill_flag_audit <- conference_data |>
  filter(skill == "Attack", !is.na(winning_attack),
         winning_attack != (evaluation_code %in% "#")) |>
  select(all_of(keys), file_line_number, evaluation_code, winning_attack)

event_flags <- conference_data |>
  filter(team == home_team | team == visiting_team) |>
  mutate(
    side = if_else(team == home_team, "home", "away"),
    attack_attempts = as.integer(skill %in% "Attack"),
    kills = as.integer(skill %in% "Attack" & evaluation_code %in% "#"),
    attack_errors = as.integer(skill %in% "Attack" & evaluation_code %in% "="),
    blocked_attacks = as.integer(skill %in% "Attack" & evaluation_code %in% "/"),
    serve_attempts = as.integer(skill %in% "Serve"),
    aces = as.integer(skill %in% "Serve" & evaluation_code %in% "#"),
    serve_errors = as.integer(skill %in% "Serve" & evaluation_code %in% "="),
    reception_attempts = as.integer(skill %in% "Reception"),
    perfect_receptions = as.integer(skill %in% "Reception" & evaluation_code %in% "#"),
    reception_errors = as.integer(skill %in% "Reception" & evaluation_code %in% "="),
    winning_blocks = as.integer(skill %in% "Block" & evaluation_code %in% "#"),
    recorded_errors = as.integer(!is.na(skill) & evaluation_code %in% "="),
    earned_points = kills + aces + winning_blocks
  )
count_names <- c("attack_attempts", "kills", "attack_errors", "blocked_attacks",
                 "serve_attempts", "aces", "serve_errors", "reception_attempts",
                 "perfect_receptions", "reception_errors", "winning_blocks",
                 "recorded_errors", "earned_points")

event_counts <- event_flags |>
  inner_join(select(point_records, all_of(keys),
                    point_record_line = file_line_number), by = keys) |>
  filter(file_line_number <= point_record_line) |>
  group_by(across(all_of(keys)), side) |>
  summarise(across(all_of(count_names), sum), .groups = "drop") |>
  pivot_wider(names_from = side, values_from = all_of(count_names),
              names_glue = "{side}_{.value}", values_fill = 0)

rally_metrics <- point_records |>
  select(all_of(keys), file_line_number, home_team, visiting_team,
         home_team_score, visiting_team_score, point_won_by, serving_team) |>
  left_join(event_counts, by = keys) |>
  inner_join(filter(set_audit, usable), by = c("match_id", "set_number")) |>
  arrange(match_id, set_number, file_line_number) |>
  group_by(match_id, set_number) |>
  mutate(
    rally_number = row_number(),
    home_received = as.integer(serving_team == visiting_team),
    away_received = as.integer(serving_team == home_team),
    home_sideouts = as.integer(home_received == 1 & point_won_by == home_team),
    away_sideouts = as.integer(away_received == 1 & point_won_by == visiting_team),
    home_served = away_received, away_served = home_received,
    home_serving_points = as.integer(home_served == 1 & point_won_by == home_team),
    away_serving_points = as.integer(away_served == 1 & point_won_by == visiting_team),
    across(all_of(names(event_counts)[!names(event_counts) %in% keys]),
           ~ cumsum(replace_na(.x, 0))),
    across(c(home_received, away_received, home_sideouts, away_sideouts,
             home_served, away_served, home_serving_points, away_serving_points),
           ~ cumsum(.x)),
    score_difference = home_team_score - visiting_team_score,
    total_points = home_team_score + visiting_team_score,
    next_home_serves = as.integer(point_won_by == home_team),
    fifth_set = as.integer(set_number == 5),
    set_over = pmax(home_team_score, visiting_team_score) >= target &
      abs(score_difference) >= 2
  ) |>
  ungroup()

# Raw rates are NA until an attempt occurs. Model rates use (successes + .5)
# / (attempts + 1), a fixed smoothing rule using no future information.
rate_definitions <- list(
  kill_pct = c("kills", "attack_attempts"),
  attack_error_pct = c("attack_errors", "attack_attempts"),
  blocked_attack_pct = c("blocked_attacks", "attack_attempts"),
  ace_pct = c("aces", "serve_attempts"),
  service_error_pct = c("serve_errors", "serve_attempts"),
  perfect_reception_pct = c("perfect_receptions", "reception_attempts"),
  reception_error_pct = c("reception_errors", "reception_attempts"),
  sideout_pct = c("sideouts", "received"),
  serving_point_pct = c("serving_points", "served")
)
for (side in c("home", "away")) {
  for (metric in names(rate_definitions)) {
    numerator <- rally_metrics[[paste0(side, "_", rate_definitions[[metric]][1])]]
    denominator <- rally_metrics[[paste0(side, "_", rate_definitions[[metric]][2])]]
    rally_metrics[[paste0(side, "_", metric)]] <-
      ifelse(denominator > 0, 100 * numerator / denominator, NA_real_)
    rally_metrics[[paste0(side, "_", metric, "_smooth")]] <-
      (numerator + 0.5) / (denominator + 1)
  }
  attempts <- rally_metrics[[paste0(side, "_attack_attempts")]]
  failures <- rally_metrics[[paste0(side, "_attack_errors")]] +
    rally_metrics[[paste0(side, "_blocked_attacks")]]
  rally_metrics[[paste0(side, "_attack_failure_pct")]] <-
    ifelse(attempts > 0, 100 * failures / attempts, NA_real_)
  rally_metrics[[paste0(side, "_attack_efficiency")]] <- ifelse(
    attempts > 0, (rally_metrics[[paste0(side, "_kills")]] - failures) / attempts,
    NA_real_)
}
for (metric in names(rate_definitions)) {
  rally_metrics[[paste0(metric, "_diff")]] <-
    rally_metrics[[paste0("home_", metric, "_smooth")]] -
    rally_metrics[[paste0("away_", metric, "_smooth")]]
}

# Do not train on the final rally, when the winner is already known.
model_data <- rally_metrics |>
  filter(!set_over) |>
  group_by(match_id, set_number) |>
  mutate(set_weight = 1 / n()) |>
  ungroup()
set.seed(4640)
match_ids <- sort(unique(model_data$match_id))
test_ids <- sample(match_ids, max(1, floor(length(match_ids) * 0.2)))
train_data <- filter(model_data, !match_id %in% test_ids)
test_data <- filter(model_data, match_id %in% test_ids)
stopifnot(length(unique(train_data$home_set_win)) == 2,
          !any(unique(train_data$match_id) %in% unique(test_data$match_id)))

# Equal total weight per set prevents longer sets dominating the fits.
score_formula <- home_set_win ~ score_difference * total_points +
  next_home_serves + fifth_set
skill_formula <- update(score_formula, . ~ . + kill_pct_diff +
  attack_error_pct_diff + blocked_attack_pct_diff + ace_pct_diff +
  service_error_pct_diff + perfect_reception_pct_diff + reception_error_pct_diff)
score_model <- glm(score_formula, data = train_data, weights = set_weight,
                   family = quasibinomial())
skill_model <- glm(skill_formula, data = train_data, weights = set_weight,
                   family = quasibinomial())
# Quasibinomial uses the logistic mean with fractional weights. Ordinary GLM
# standard errors do not handle repeated rallies; do not use them for inference.

test_predictions <- test_data |>
  mutate(score_only = predict(score_model, newdata = test_data, type = "response"),
         score_plus_skills = predict(skill_model, newdata = test_data, type = "response")) |>
  pivot_longer(c(score_only, score_plus_skills), names_to = "model",
               values_to = "win_probability") |>
  mutate(
    brier = (win_probability - home_set_win)^2,
    clipped_probability = pmin(pmax(win_probability, 1e-8), 1 - 1e-8),
    log_loss = -(home_set_win * log(clipped_probability) +
                   (1 - home_set_win) * log(1 - clipped_probability)),
    stage = case_when(total_points <= 15 ~ "Early: 1–15 points",
                      total_points <= 30 ~ "Middle: 16–30 points",
                      TRUE ~ "Late: 31+ points")
  )
model_comparison <- test_predictions |>
  group_by(model, match_id, set_number) |>
  summarise(brier = mean(brier), log_loss = mean(log_loss), .groups = "drop") |>
  group_by(model) |>
  summarise(test_sets = n(), brier = mean(brier), log_loss = mean(log_loss),
            .groups = "drop")
stage_comparison <- test_predictions |>
  group_by(model, stage, match_id, set_number) |>
  summarise(brier = mean(brier), .groups = "drop") |>
  group_by(model, stage) |>
  summarise(test_sets = n(), brier = mean(brier), .groups = "drop")
calibration_table <- test_predictions |>
  mutate(probability_bin = pmin(floor(win_probability * 10), 9L)) |>
  group_by(model, probability_bin) |>
  summarise(predicted = weighted.mean(win_probability, set_weight),
            observed = weighted.mean(home_set_win, set_weight),
            snapshots = n(), .groups = "drop")
