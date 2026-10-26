# Whole-set summaries: one row per team per set (two rows per set).
# Requires dplyr and tidyr. This file defines a function; it does not fit models.
# Rates are proportions (0 to 1), not percentages (0 to 100).

wrangle_set_skills <- function(raw_data) {
  required <- c("match_id", "set_number", "file_line_number", "point_id",
                "home_team", "visiting_team", "team", "skill",
                "evaluation_code", "point", "point_won_by",
                "home_team_score", "visiting_team_score")
  missing <- setdiff(required, names(raw_data))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
  x <- dplyr::select(raw_data, dplyr::all_of(required))
  if (anyNA(x[c("match_id", "set_number", "file_line_number",
                "home_team", "visiting_team")])) stop("Missing identifiers or team names.")
  if (any(!nzchar(x$match_id)) || any(!x$set_number %in% 1:5))
    stop("Invalid match IDs or set numbers; this function assumes best-of-five matches.")
  teams <- dplyr::distinct(x, match_id, home_team, visiting_team)
  if (anyDuplicated(teams$match_id) || any(teams$home_team == teams$visiting_team))
    stop("Each match must have one distinct home and visiting team.")
  if (anyDuplicated(x[c("match_id", "file_line_number")]))
    stop("Duplicate event lines within a match; inspect before aggregating.")

  # 1. Point records contain the score after each rally. Administrative
  # end-of-set markers can belong to the next set, so do not use their scores.
  points <- x |>
    dplyr::filter(point %in% TRUE) |>
    dplyr::arrange(match_id, set_number, file_line_number)
  if (anyNA(points$point_id) ||
      anyDuplicated(points[c("match_id", "set_number", "point_id")]))
    stop("Missing or duplicated rally identifiers in point records.")

  results <- points |>
    dplyr::group_by(match_id, set_number) |>
    dplyr::summarise(
      final_home_score = dplyr::last(home_team_score),
      final_visiting_score = dplyr::last(visiting_team_score),
      recorded_points = dplyr::n(),
      score_sequence_valid = all(
        !is.na(home_team_score) & !is.na(visiting_team_score) &
          home_team_score >= 0 & visiting_team_score >= 0) &
        all(diff(c(0, home_team_score)) %in% 0:1) &
        all(diff(c(0, visiting_team_score)) %in% 0:1) &
        all(home_team_score + visiting_team_score == dplyr::row_number()),
      point_winners_valid = all(point_won_by == ifelse(
        home_team_score > dplyr::lag(home_team_score, default = 0),
        home_team, visiting_team)),
      no_early_finish = !any(head(
        pmax(home_team_score, visiting_team_score) >=
          ifelse(dplyr::first(set_number) == 5, 15, 25) &
          abs(home_team_score - visiting_team_score) >= 2, -1)),
      .groups = "drop"
    )
  set_results <- x |>
    dplyr::distinct(match_id, set_number, home_team, visiting_team) |>
    dplyr::left_join(results, by = c("match_id", "set_number")) |>
    dplyr::mutate(
      set_complete = dplyr::coalesce(
        pmax(final_home_score, final_visiting_score) >=
          ifelse(set_number == 5, 15, 25) &
          abs(final_home_score - final_visiting_score) >= 2, FALSE),
      score_sequence_valid = dplyr::coalesce(score_sequence_valid, FALSE),
      point_winners_valid = dplyr::coalesce(point_winners_valid, FALSE),
      no_early_finish = dplyr::coalesce(no_early_finish, FALSE),
      scoring_valid = set_complete & score_sequence_valid &
        point_winners_valid & no_early_finish,
      winning_team = dplyr::case_when(
        set_complete & final_home_score > final_visiting_score ~ home_team,
        set_complete ~ visiting_team,
        TRUE ~ NA_character_)
    )

  # 2. Keep skill actions only. Preserve unexpected records for inspection.
  # The supplied CSV uses 'Freeball'; accept the teammate spelling as well.
  skills <- c("Attack", "Serve", "Reception", "Block", "Set", "Dig", "Freeball")
  actions <- x |>
    dplyr::mutate(skill = dplyr::recode(skill, "Free Ball" = "Freeball")) |>
    dplyr::filter(!is.na(skill), skill != "Timeout") |>
    dplyr::mutate(valid_action = skill %in% skills & !is.na(team) &
      (team == home_team | team == visiting_team) &
      evaluation_code %in% c("#", "=", "/", "!", "+", "-"))
  issues <- dplyr::filter(actions, !valid_action)
  counts <- actions |>
    dplyr::filter(valid_action) |>
    dplyr::group_by(match_id, set_number, team) |>
    dplyr::summarise(
      attack_attempts = sum(skill == "Attack"),
      attack_kills = sum(skill == "Attack" & evaluation_code == "#"),
      attack_errors = sum(skill == "Attack" & evaluation_code == "="),
      blocked_attacks = sum(skill == "Attack" & evaluation_code == "/"),
      serve_attempts = sum(skill == "Serve"),
      aces = sum(skill == "Serve" & evaluation_code == "#"),
      serve_errors = sum(skill == "Serve" & evaluation_code == "="),
      reception_attempts = sum(skill == "Reception"),
      perfect_receptions = sum(skill == "Reception" & evaluation_code == "#"),
      reception_errors = sum(skill == "Reception" & evaluation_code == "="),
      winning_blocks = sum(skill == "Block" & evaluation_code == "#"),
      set_errors = sum(skill == "Set" & evaluation_code == "="),
      freeball_errors = sum(skill == "Freeball" & evaluation_code == "="),
      .groups = "drop"
    )
  count_columns <- setdiff(names(counts), c("match_id", "set_number", "team"))
  issue_counts <- issues |>
    dplyr::count(match_id, set_number, name = "invalid_skill_rows")
  set_results <- set_results |>
    dplyr::left_join(issue_counts, by = c("match_id", "set_number")) |>
    dplyr::mutate(invalid_skill_rows = tidyr::replace_na(invalid_skill_rows, 0L),
                  usable = scoring_valid & invalid_skill_rows == 0)

  # 3. Explicitly create both team rows, then attach counts and rates.
  # Undefined ratios remain NA; zero errors are represented by a separate flag.
  divide <- function(numerator, denominator) {
    ifelse(denominator > 0, numerator / denominator, NA_real_)
  }
  team_sets <- set_results |>
    tidyr::pivot_longer(c(home_team, visiting_team),
                        names_to = "side", values_to = "team") |>
    dplyr::left_join(counts, by = c("match_id", "set_number", "team")) |>
    dplyr::mutate(
      dplyr::across(dplyr::all_of(count_columns), ~ tidyr::replace_na(.x, 0L)),
      is_home = side == "home_team",
      score_for = ifelse(is_home, final_home_score, final_visiting_score),
      score_against = ifelse(is_home, final_visiting_score, final_home_score),
      set_won = factor(ifelse(is.na(winning_team), NA_character_,
                             ifelse(team == winning_team, "win", "loss")),
                       levels = c("loss", "win")),
      kill_rate = divide(attack_kills, attack_attempts),
      attack_error_rate = divide(attack_errors, attack_attempts),
      attack_failure_rate = divide(attack_errors + blocked_attacks, attack_attempts),
      # Explicit definition: blocked attacks count as failures here; ! does not.
      attack_efficiency = divide(attack_kills - attack_errors - blocked_attacks,
                                 attack_attempts),
      ace_rate = divide(aces, serve_attempts),
      serve_error_rate = divide(serve_errors, serve_attempts),
      perfect_reception_rate = divide(perfect_receptions, reception_attempts),
      reception_error_rate = divide(reception_errors, reception_attempts),
      attack_point_error_ratio = divide(attack_kills, attack_errors),
      serve_point_error_ratio = divide(aces, serve_errors),
      no_attack_errors = attack_errors == 0,
      no_serve_errors = serve_errors == 0
    ) |>
    dplyr::arrange(match_id, set_number, side)

  # Retain flagged sets so exclusions are visible and reviewable.
  # 'usable' checks recorded scoring/coding, not the accuracy of every scout code.
  list(team_sets = team_sets, set_audit = set_results, skill_issues = issues)
}
