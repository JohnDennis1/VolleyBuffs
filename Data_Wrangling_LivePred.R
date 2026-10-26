# Wrangling meant for the live prediction
# Run from the project root: Rscript R/build_live_outputs.R
# No other scripts are needed necessarily, but it does require dplyr and tidyr.
#
# Starts with exclusions -> columns -> skill indicators ->
# cumulative counts -> rates -> two modeling tables -> save outputs.
# One output row describes the state AFTER a completed, non-final rally.
# Points come from point records, never from adding skill successes and errors in order to avoid double counting.
# No models, data splits, smoothing, scaling, or feature selection are fitted.
# Assumes the current conference export, already inspected, with three agreed exclusions.
#
# Just to clarify "<-" assigns; "|>" pipe operator; "::" allows for package use;
# "$" extracts a named item; "[[name]]" extracts a column using a stored name;
# "%in%" checks whether it is part of the condition; 
# "as.integer" converts TRUE/FALSE indicators to 1/0.

# Change these two paths if your team's files live elsewhere.
input_file <- "Data/conference_matches_raw.csv"
output_dir <- "Data/derived"

library(dplyr)
library(tidyr)
raw_data <- read.csv(input_file)

# Removing these sets for the reason listed thereafter. I am 90% sure that their absences wouldn't be that impactful. 
excluded_sets <- tibble::tribble(
  ~match_id, ~set_number, ~reason,
  "c03b4acd4e438538b3fff7193f9404ca", 3L, "Missing or invalid serving information",
  "2fae8145af5d1bf7865c49bfacddb5db", 1L, "Serving sequence inconsistent with preceding point winner",
  "3c644ccf406fb8127d5bf244d46adb83", 2L, "Invalid scoring sequence"
)

# Both keys matter: remove the affected set, not the entire match.
clean_data <- anti_join(raw_data, excluded_sets, by = c("match_id", "set_number"))
# Step 2: keep only fields needed to build/check rally snapshots and interpret
# skill codes. Notes/codes remain here for inspection, not as model predictors.
source_columns <- c(
  "match_id", "set_number", "point_id", "file_line_number",
  "home_team", "visiting_team", "team",
  "point", "point_won_by", "serving_team",
  "home_team_score", "visiting_team_score",
  "skill", "evaluation_code", "evaluation",
  "special_code", "custom_code", "code"
)
live_events <- select(clean_data, all_of(source_columns))

# 3. Identify rallies. Together, these three fields uniquely identify a rally.
keys <- c("match_id", "set_number", "point_id")
# A TRUE point row records the score after a rally; arrange restores event order.
point_records <- live_events |>
  filter(point %in% TRUE) |>
  arrange(match_id, set_number, file_line_number)
# The last recorded score supplies the historical outcome for each cleaned set.
# This label is kept separate from the model's predictors.
set_results <- point_records |>
  group_by(match_id, set_number) |>
  summarise(
    target = if_else(first(set_number) == 5, 15L, 25L),
    home_set_win = as.integer(last(home_team_score) > last(visiting_team_score)),
    .groups = "drop"
  )

# Keep skill actions, and attach the finishing line of their completed rally.
# The line cutoff ensures we use only events available at that prediction time.
actions <- live_events |>
  filter(!is.na(skill), skill != "Timeout") |>
  inner_join(select(point_records, all_of(keys),
                    point_record_line = file_line_number), by = keys) |>
  filter(file_line_number <= point_record_line)

# 4. Turn each recorded action into 0/1 indicators. as.integer(TRUE) is 1.
event_flags <- actions |>
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
    # Attempts are recorded touches, not numbers of rallies.
    block_attempts = as.integer(skill %in% "Block"),
    block_errors = as.integer(skill %in% "Block" & evaluation_code %in% "="),
    dig_attempts = as.integer(skill %in% "Dig"),
    dig_errors = as.integer(skill %in% "Dig" & evaluation_code %in% "="),
    set_attempts = as.integer(skill %in% "Set"),
    set_errors = as.integer(skill %in% "Set" & evaluation_code %in% "="),
    freeball_attempts = as.integer(skill %in% "Freeball"),
    freeball_errors = as.integer(skill %in% "Freeball" & evaluation_code %in% "="),
    # This is our provisional project definition, not a decoded special code.
    unforced_errors = attack_errors + serve_errors,
    # Forced is provisional too; Block/= is not necessarily an illegal block.
    forced_error_actions = blocked_attacks + block_errors + dig_errors,
    unclassified_errors = reception_errors + set_errors + freeball_errors,
    # These are successful skill actions. NEVER add them to opponents' errors
    # to reconstruct points: an ace and a reception error can share one rally.
    winning_skill_actions = kills + aces + winning_blocks,
    error_class = case_when(
      unforced_errors == 1 ~ "unforced_provisional",
      forced_error_actions == 1 ~ "forced_provisional",
      unclassified_errors == 1 ~ "unclassified",
      TRUE ~ "not_classified_as_error"
    )
  )
# List just the columns to sum; text labels are not summed.
count_names <- c("attack_attempts", "kills", "attack_errors", "blocked_attacks",
                 "serve_attempts", "aces", "serve_errors", "reception_attempts",
                 "perfect_receptions", "reception_errors", "winning_blocks",
                 "block_attempts", "block_errors", "dig_attempts", "dig_errors",
                 "set_attempts", "set_errors", "freeball_attempts", "freeball_errors",
                 "unforced_errors", "forced_error_actions", "unclassified_errors",
                 "winning_skill_actions")

# Sum indicators within rally and team, then put home/away counts side by side.
event_counts <- event_flags |>
  group_by(across(all_of(keys)), side) |>
  summarise(across(all_of(count_names), sum), .groups = "drop") |>
  pivot_wider(names_from = side, values_from = all_of(count_names),
              names_glue = "{side}_{.value}", values_fill = 0)

# 5. Attach skill counts to point records and accumulate within each set.
rally_metrics <- point_records |>
  select(all_of(keys), file_line_number, home_team, visiting_team,
         home_team_score, visiting_team_score, point_won_by, serving_team) |>
  left_join(event_counts, by = keys) |>
  inner_join(set_results, by = c("match_id", "set_number")) |>
  arrange(match_id, set_number, file_line_number) |>
  group_by(match_id, set_number) |>
  mutate(
    # row_number restarts because the data are grouped by match and set.
    rally_number = row_number(),
    # Receiving means the opponent served THIS completed rally.
    home_received = as.integer(serving_team == visiting_team),
    away_received = as.integer(serving_team == home_team),
    # A sideout is winning a rally while receiving.
    home_sideouts = as.integer(home_received == 1 & point_won_by == home_team),
    away_sideouts = as.integer(away_received == 1 & point_won_by == visiting_team),
    home_served = away_received, away_served = home_received,
    home_serving_points = as.integer(home_served == 1 & point_won_by == home_team),
    away_serving_points = as.integer(away_served == 1 & point_won_by == visiting_team),
    # Replace absent recorded actions with zero; cumsum accumulates in order.
    across(all_of(names(event_counts)[!names(event_counts) %in% keys]),
           ~ cumsum(replace_na(.x, 0))),
    # Accumulate serving/receiving opportunities and wins separately.
    across(c(home_received, away_received, home_sideouts, away_sideouts,
             home_served, away_served, home_serving_points, away_serving_points),
           ~ cumsum(.x)),
    # Points come only from rally winners, independently of the skill counts.
    home_points_won = cumsum(as.integer(point_won_by == home_team)),
    away_points_won = cumsum(as.integer(point_won_by == visiting_team)),
    score_difference = home_team_score - visiting_team_score,
    total_points = home_team_score + visiting_team_score,
    next_home_serves = as.integer(point_won_by == home_team),
    fifth_set = as.integer(set_number == 5),
    set_over = pmax(home_team_score, visiting_team_score) >= target &
      abs(score_difference) >= 2
  ) |>
  ungroup()

# Raw rates remain NA until an attempt occurs. Counts preserve exposure.
# No automatic smoothing: a zero-attempt ace rate is not assumed to be 50%.
# 6. Each pair names the cumulative numerator and denominator for a rate.
rate_definitions <- list(
  kill_pct = c("kills", "attack_attempts"),
  attack_error_pct = c("attack_errors", "attack_attempts"),
  blocked_attack_pct = c("blocked_attacks", "attack_attempts"),
  ace_pct = c("aces", "serve_attempts"),
  service_error_pct = c("serve_errors", "serve_attempts"),
  perfect_reception_pct = c("perfect_receptions", "reception_attempts"),
  reception_error_pct = c("reception_errors", "reception_attempts"),
  block_win_pct = c("winning_blocks", "block_attempts"),
  block_error_pct = c("block_errors", "block_attempts"),
  dig_error_pct = c("dig_errors", "dig_attempts"),
  set_error_pct = c("set_errors", "set_attempts"),
  freeball_error_pct = c("freeball_errors", "freeball_attempts"),
  sideout_pct = c("sideouts", "received"),
  serving_point_pct = c("serving_points", "served")
)
for (side in c("home", "away")) {
  for (metric in names(rate_definitions)) {
    # [[ ]] extracts a column whose name is assembled with paste0.
    numerator <- rally_metrics[[paste0(side, "_", rate_definitions[[metric]][1])]]
    denominator <- rally_metrics[[paste0(side, "_", rate_definitions[[metric]][2])]]
    # Return a percentage only when its denominator is positive.
    rally_metrics[[paste0(side, "_", metric)]] <-
      ifelse(denominator > 0, 100 * numerator / denominator, NA_real_)
  }
  # Ratios are retained for comparison with the whole-set wrangler. Zero errors
  # mean an undefined ratio, not zero success; the flags retain that information.
  errors <- rally_metrics[[paste0(side, "_attack_errors")]]
  serve_errors <- rally_metrics[[paste0(side, "_serve_errors")]]
  rally_metrics[[paste0(side, "_attack_point_error_ratio")]] <-
    ifelse(errors > 0, rally_metrics[[paste0(side, "_kills")]] / errors, NA_real_)
  rally_metrics[[paste0(side, "_serve_point_error_ratio")]] <-
    ifelse(serve_errors > 0, rally_metrics[[paste0(side, "_aces")]] / serve_errors, NA_real_)
  rally_metrics[[paste0(side, "_no_attack_errors")]] <- as.integer(errors == 0)
  rally_metrics[[paste0(side, "_no_serve_errors")]] <- as.integer(serve_errors == 0)
  attempts <- rally_metrics[[paste0(side, "_attack_attempts")]]
  failures <- rally_metrics[[paste0(side, "_attack_errors")]] +
    rally_metrics[[paste0(side, "_blocked_attacks")]]
  rally_metrics[[paste0(side, "_attack_failure_pct")]] <-
    ifelse(attempts > 0, 100 * failures / attempts, NA_real_)
  rally_metrics[[paste0(side, "_attack_efficiency")]] <- ifelse(
    attempts > 0, (rally_metrics[[paste0(side, "_kills")]] - failures) / attempts,
    NA_real_)
}
# Explicit names distinguish percentage-point differences from former smoothed
# proportion differences; old *_diff predictors must not silently change units.
for (metric in names(rate_definitions)) {
  rally_metrics[[paste0(metric, "_raw_diff")]] <-
    rally_metrics[[paste0("home_", metric)]] - rally_metrics[[paste0("away_", metric)]]
}

# Do not train on the final rally, when the winner is already known.
# Remove final rallies; weights give each retained set a total weight of one.
model_data <- rally_metrics |>
  filter(!set_over) |>
  group_by(match_id, set_number) |>
  mutate(set_weight = 1 / n()) |>
  ungroup()

# 7. Produce two tables with identical rows and clearly separated column roles.
# Metadata lets us trace predictions and split by MATCH. It is never an input.
metadata_columns <- c(keys, "file_line_number", "home_team", "visiting_team",
                      "rally_number", "home_team_score", "visiting_team_score",
                      "target", "home_set_win", "set_weight")
# These four inputs define the common score-only baseline for every algorithm.
score_predictors <- c("score_difference", "total_points", "next_home_serves", "fifth_set")
# Include all constructed skill features in the candidate table, but do not
# silently feed every redundant count/rate/ratio to a model.
count_columns <- as.vector(outer(c("home", "away"), count_names, paste, sep = "_"))
rate_columns <- as.vector(outer(c("home", "away"), names(rate_definitions), paste, sep = "_"))
extra_metrics <- c("attack_failure_pct", "attack_efficiency", "attack_point_error_ratio",
                   "serve_point_error_ratio", "no_attack_errors", "no_serve_errors",
                   "received", "served", "sideouts", "serving_points")
extra_columns <- as.vector(outer(c("home", "away"), extra_metrics, paste, sep = "_"))
skill_candidates <- c(count_columns, rate_columns, extra_columns,
                      paste0(names(rate_definitions), "_raw_diff"))
score_only <- as.data.frame(select(model_data,
  all_of(c(metadata_columns, score_predictors))))
score_plus_skills <- as.data.frame(select(model_data,
  all_of(c(metadata_columns, score_predictors, skill_candidates))))

# A transparent starting feature set uses counts (no undefined rates). It is
# not a claim these are the best predictors. Aggregate errors/successes and
# their components are not all included simultaneously. Unforced totals remain
# available in score_plus_skills for an explicitly chosen alternative model.
base_counts <- c("attack_attempts", "kills", "attack_errors", "blocked_attacks",
  "serve_attempts", "aces", "serve_errors", "reception_attempts",
  "perfect_receptions", "reception_errors", "block_attempts", "winning_blocks",
  "block_errors", "dig_attempts", "dig_errors", "set_attempts", "set_errors",
  "freeball_attempts", "freeball_errors")
skill_predictors <- c(score_predictors,
  as.vector(outer(c("home", "away"), base_counts, paste, sep = "_")))
# Numeric matrices contain inputs ONLY. Keep y and metadata separate.
x_score <- as.matrix(score_only[score_predictors])
x_skills <- as.matrix(score_plus_skills[skill_predictors])
# Explicit factor levels make loss/win classification consistent across models.
y <- factor(model_data$home_set_win, levels = c(0, 1), labels = c("loss", "win"))
# Check that paired skill records have not increased the number of points.
stopifnot(all(rally_metrics$home_points_won == rally_metrics$home_team_score),
          all(rally_metrics$away_points_won == rally_metrics$visiting_team_score),
          all(rally_metrics$home_points_won + rally_metrics$away_points_won ==
                rally_metrics$rally_number),
          identical(score_only[keys], score_plus_skills[keys]),
          all(is.finite(x_score)), all(is.finite(x_skills)))

# Collect all outputs in one named list for saving and later model implementation.
live <- list(score_only = score_only, score_plus_skills = score_plus_skills,
     x_score = x_score, x_skills = x_skills, y = y,
     score_predictors = score_predictors, skill_predictors = skill_predictors,
     skill_candidates = skill_candidates,
     # Event details retain special/custom codes without using them as inputs.
     event_details = event_flags,
     excluded_sets = excluded_sets, rate_definitions = rate_definitions)

# 8. Save outputs only after the calculations and consistency checks succeed.
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(live_events, file.path(output_dir, "live_events_clean.csv"), row.names = FALSE)
write.csv(excluded_sets, file.path(output_dir, "live_excluded_sets.csv"), row.names = FALSE)
# Save the full bundle in R's format, preserving matrices and factor levels.
saveRDS(live, file.path(output_dir, "live_analysis.rds"))
# CSVs are convenient for inspection; use the RDS when preserving R types matters.
write.csv(score_only, file.path(output_dir, "live_score_only.csv"), row.names = FALSE)
write.csv(score_plus_skills, file.path(output_dir, "live_score_plus_skills.csv"), row.names = FALSE)
# Print dimensions so readers can verify the two tables have the same row count.
cat("Score-only table:", nrow(score_only), "rows;", ncol(score_only), "columns.\n")
cat("Score-plus-skills table:", nrow(score_plus_skills), "rows;", ncol(score_plus_skills), "columns.\n")
cat("Starting matrices:", ncol(x_score), "score inputs and", ncol(x_skills), "score/skill inputs.\n")
