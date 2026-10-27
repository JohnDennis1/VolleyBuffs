# THIS was Updated as of OCT 27 15:20
# Everything here should run within the proper data pipeline.

#Data/conference_matches_raw.csv -> Data_Wrangling_LivePred.R -> Data/derived/live_analysis.rds
# -> make_splits.R -> model_splits.csv

# Historical live-prediction data: one row AFTER each completed rally.
# Sourcing this file only defines a function. No models or data splits are run.
# Call: live <- wrangle_live_skills(conference_data)
# Start with R/build_live_outputs.R; see LIVE_WRANGLING.md for an R syntax guide.
# Use live$score_only and live$score_plus_skills; select explicit predictors.
# Rates are raw percentages (0--100); efficiency is a proportion (-1--1).
# Smoothing/scaling/imputation belong in training-only preprocessing, not here.
# This is a historical dataset builder, not an ongoing-match prediction API.
wrangle_live_skills <- function(conference_data) {
  if (!requireNamespace("dplyr", quietly = TRUE) ||
      !requireNamespace("tidyr", quietly = TRUE)) stop("Install dplyr and tidyr.")
  # These columns must exist; optional annotation columns are retained if supplied.
  required <- c("match_id", "set_number", "point_id", "file_line_number",
    "home_team", "visiting_team", "team", "skill", "evaluation_code", "point",
    "point_won_by", "serving_team", "home_team_score", "visiting_team_score")
  # setdiff lists required column names absent from the input.
  missing <- setdiff(required, names(conference_data))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
  # Work on a local copy; the caller's original data is not modified.
  x <- conference_data
  if (anyNA(x[c("match_id", "set_number", "file_line_number", "home_team", "visiting_team")]))
    stop("Missing identifiers or team names.")
  if (any(!nzchar(x$match_id)) || any(!x$set_number %in% 1:5))
    stop("Invalid match IDs or set numbers; expected best-of-five matches.")
  if (!is.numeric(x$file_line_number) || any(!is.finite(x$file_line_number)))
    stop("file_line_number must be finite numeric values.")
  if (anyDuplicated(x[c("match_id", "file_line_number")]))
    stop("Duplicate source lines within a match.")
  # A match must refer to one consistent pair of different teams.
  teams <- dplyr::distinct(x, match_id, home_team, visiting_team)
  if (anyDuplicated(teams$match_id) || any(teams$home_team == teams$visiting_team) ||
      any(!nzchar(teams$home_team)) || any(!nzchar(teams$visiting_team)))
    stop("Each match must have one distinct home and visiting team.")
  if (!is.logical(x$point) || anyNA(x$point)) stop("point must be nonmissing logical values.")
  if (!is.numeric(x$home_team_score) || !is.numeric(x$visiting_team_score))
    stop("Scores must be numeric.")
  if (!"winning_attack" %in% names(x)) x$winning_attack <- NA
  x$skill[x$skill %in% "Free Ball"] <- "Freeball"
  conference_data <- x

  # 1. Identify rallies. Together, these three fields uniquely identify a rally.
  keys <- c("match_id", "set_number", "point_id")
  # A TRUE point row records the score after a rally; arrange restores event order.
  point_records <- conference_data |>
    dplyr::filter(point %in% TRUE) |>
    dplyr::arrange(match_id, set_number, file_line_number)
  if (anyNA(point_records$point_id) || anyDuplicated(point_records[keys]))
    stop("Missing or duplicated rally identifiers in point records.")

  # Audit complete scoring sequences before using a set for model development.
  # 2. Validate sets before using their eventual winner as a training label.
  set_audit <- point_records |>
    dplyr::group_by(match_id, set_number) |>
    dplyr::summarise(
      # Sets 1--4 target 25; the deciding fifth set targets 15.
      target = dplyr::if_else(dplyr::first(set_number) == 5, 15L, 25L),
      # A finishing score must reach the target AND lead by at least two.
      complete = dplyr::last(pmax(home_team_score, visiting_team_score)) >= target &
        abs(dplyr::last(home_team_score - visiting_team_score)) >= 2,
      # Scores start at zero, increase by 0/1 per team, and add to rally number.
      scores_valid = all(!is.na(home_team_score) & !is.na(visiting_team_score) &
        home_team_score >= 0 & visiting_team_score >= 0) &
        all(diff(c(0, home_team_score)) %in% 0:1) &
        all(diff(c(0, visiting_team_score)) %in% 0:1) &
        all(home_team_score + visiting_team_score == dplyr::row_number()),
      # The named rally winner must match whichever score increased.
      winners_valid = all(dplyr::coalesce(point_won_by == dplyr::if_else(
        home_team_score > dplyr::lag(home_team_score, default = 0L),
        home_team, visiting_team), FALSE)),
      # head(..., -1) drops the last rally; none before it may end the set.
      no_early_finish = !any(head(
        pmax(home_team_score, visiting_team_score) >= target &
          abs(home_team_score - visiting_team_score) >= 2, -1)),
      # The server must be one of the two teams, with no missing values.
      serving_valid = all(!is.na(serving_team) &
        (serving_team == home_team | serving_team == visiting_team)),
      # After rally one, the server must equal the preceding rally winner.
      serving_sequence_valid = all(dplyr::coalesce(
        serving_team == dplyr::lag(point_won_by), dplyr::row_number() == 1)),
      # Historical outcome label ONLY; never put it in a predictor matrix.
      home_set_win = as.integer(dplyr::last(home_team_score) > dplyr::last(visiting_team_score)),
      .groups = "drop"
    ) |>
    dplyr::mutate(usable = dplyr::coalesce(complete & scores_valid & winners_valid &
                              no_early_finish & serving_valid & serving_sequence_valid, FALSE))

  # Include sets with no point records in the audit. A missing final rally is
  # incomplete recording, not a draw. Never invent an outcome from a current lead.
  set_audit <- dplyr::distinct(conference_data, match_id, set_number) |>
    dplyr::left_join(set_audit, by = c("match_id", "set_number")) |>
    dplyr::mutate(target = dplyr::if_else(set_number == 5, 15L, 25L),
           usable = dplyr::coalesce(usable, FALSE),
           complete = dplyr::coalesce(complete, FALSE),
           home_set_win = dplyr::if_else(usable, home_set_win, NA_integer_))

  # Audit skill coding and rally linkage before counting. Administrative rows
  # and timeouts are not skill actions. Bad actions exclude their set by default.
  actions <- conference_data |>
    dplyr::filter(!is.na(skill), skill != "Timeout") |>
    dplyr::left_join(dplyr::select(point_records, dplyr::all_of(keys),
                     point_record_line = file_line_number), by = keys) |>
    dplyr::mutate(valid_code = skill %in% c("Attack", "Serve", "Reception", "Block",
                                    "Set", "Dig", "Freeball") &
             evaluation_code %in% c("#", "=", "/", "!", "+", "-"),
           valid_team = !is.na(team) & (team == home_team | team == visiting_team),
           linked = !is.na(point_record_line),
           within_rally = dplyr::coalesce(file_line_number <= point_record_line, FALSE),
           valid_action = valid_code & valid_team & linked & within_rally)
  skill_issues <- dplyr::filter(actions, !valid_action)
  # Count bad actions per set so exclusion reasons remain inspectable.
  issue_counts <- dplyr::count(skill_issues, match_id, set_number, name = "invalid_skill_rows")
  set_audit <- set_audit |>
    dplyr::left_join(issue_counts, by = c("match_id", "set_number")) |>
    dplyr::mutate(invalid_skill_rows = tidyr::replace_na(invalid_skill_rows, 0L),
           usable = usable & invalid_skill_rows == 0,
           home_set_win = dplyr::if_else(usable, home_set_win, NA_integer_))

  # A kill is defined by the Attack/# combination. Retain winning_attack in
  # the source and report disagreements rather than silently mixing definitions.
  kill_flag_audit <- conference_data |>
    dplyr::filter(skill == "Attack", !is.na(winning_attack),
           winning_attack != (evaluation_code %in% "#")) |>
    dplyr::select(dplyr::all_of(keys), file_line_number, evaluation_code, winning_attack)

  # 3. Turn each recorded action into 0/1 indicators. as.integer(TRUE) is 1.
  event_flags <- actions |>
    dplyr::filter(valid_action) |>
    dplyr::mutate(
      side = dplyr::if_else(team == home_team, "home", "away"),
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
      unforced_errors = attack_errors + serve_errors + 
        set_errors + freeball_errors,
      # Forced is provisional too; Block/= is not necessarily an illegal block.
      forced_error_actions = blocked_attacks + block_errors + dig_errors,
      unclassified_errors = reception_errors,
      # These are successful skill actions. NEVER add them to opponents' errors
      # to reconstruct points: an ace and a reception error can share one rally.
      winning_skill_actions = kills + aces + winning_blocks,
      error_class = dplyr::case_when(
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
    dplyr::group_by(dplyr::across(dplyr::all_of(keys)), side) |>
    dplyr::summarise(dplyr::across(dplyr::all_of(count_names), sum), .groups = "drop") |>
    tidyr::pivot_wider(names_from = side, values_from = dplyr::all_of(count_names),
                names_glue = "{side}_{.value}", values_fill = 0)

  for (side in c("home", "away")) {
    for (metric in count_names) {
      column <- paste0(side, "_", metric)
      if (!column %in% names(event_counts)) event_counts[[column]] <- integer(nrow(event_counts))
    }
  }

  # 4. Attach skill counts to point records and accumulate within each set.
  rally_metrics <- point_records |>
    dplyr::select(dplyr::all_of(keys), file_line_number, home_team, visiting_team,
           home_team_score, visiting_team_score, point_won_by, serving_team) |>
    dplyr::left_join(event_counts, by = keys) |>
    dplyr::inner_join(dplyr::filter(set_audit, usable), by = c("match_id", "set_number")) |>
    dplyr::arrange(match_id, set_number, file_line_number) |>
    dplyr::group_by(match_id, set_number) |>
    dplyr::mutate(
      # row_number restarts because the data are grouped by match and set.
      rally_number = dplyr::row_number(),
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
      dplyr::across(dplyr::all_of(names(event_counts)[!names(event_counts) %in% keys]),
             ~ cumsum(tidyr::replace_na(.x, 0))),
      # Accumulate serving/receiving opportunities and wins separately.
      dplyr::across(c(home_received, away_received, home_sideouts, away_sideouts,
               home_served, away_served, home_serving_points, away_serving_points),
             ~ cumsum(.x)),
      # Points come only from rally winners, independently of the skill counts.
      home_points_won = cumsum(as.integer(point_won_by == home_team)),
      away_points_won = cumsum(as.integer(point_won_by == visiting_team)),
      score_difference = home_team_score - visiting_team_score,
      total_points = home_team_score + visiting_team_score,
      next_home_serves = as.integer(point_won_by == home_team),
      fifth_set = as.integer(set_number == 5),
      # Relative progress can exceed 1 during extended sets. It is not a forecast
      # of how many rallies remain. Preserve target and fifth_set for modeling.
      score_progress = pmax(home_team_score, visiting_team_score) / target,
      deuce = home_team_score >= target - 1 & visiting_team_score >= target - 1,
      set_over = pmax(home_team_score, visiting_team_score) >= target &
        abs(score_difference) >= 2
    ) |>
    dplyr::ungroup()

  # Raw rates remain NA until an attempt occurs. Counts preserve exposure.
  # No automatic smoothing: a zero-attempt ace rate is not assumed to be 50%.
  # 5. Each pair names the cumulative numerator and denominator for a rate.
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
    dplyr::filter(!set_over) |>
    dplyr::group_by(match_id, set_number) |>
    dplyr::mutate(set_weight = 1 / dplyr::n()) |>
    dplyr::ungroup()

  # 6. Produce two tables with identical rows and clearly separated column roles.
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
  score_only <- as.data.frame(dplyr::select(model_data,
    dplyr::all_of(c(metadata_columns, score_predictors))))
  score_plus_skills <- as.data.frame(dplyr::select(model_data,
    dplyr::all_of(c(metadata_columns, score_predictors, skill_candidates))))

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

  # Return one named list; use live$score_only, live$x_skills, etc. to access items.
  list(score_only = score_only, score_plus_skills = score_plus_skills,
       x_score = x_score, x_skills = x_skills, y = y,
       score_predictors = score_predictors, skill_predictors = skill_predictors,
       skill_candidates = skill_candidates,
       # Event details retain special/custom codes without using them as inputs.
       event_details = event_flags,
       rally_metrics = rally_metrics, model_data = model_data,
       set_audit = set_audit, skill_issues = skill_issues,
       kill_flag_audit = kill_flag_audit, rate_definitions = rate_definitions)
}
