# Shared train/test split and cross-validation folds for every model.
# Run from the project root after Data_Wrangling_LivePred.R:
#   Rscript make_splits.R
# Output: model_splits.csv (committed), one row per match.
#
# Design: season holdout. The model is judged on seasons it has never seen.
#   - Test:  2023 Pac-12 season + the three 2024 Big 12 matches. Use ONCE,
#            for the final comparison only.
#   - Train: 2018-2022. Each fold is one season, so cross-validation asks
#            "trained on the other seasons, how well does it predict this one?"
#            The single 2018 match is grouped with 2019.
# All snapshots from a match share its role and fold, so a match never
# appears on both sides of a split.
#
# The split is committed as a file, not regenerated from a seed, so every
# teammate uses exactly the same matches regardless of R version.
#
# Usage in a model file:
#   splits <- read.csv("model_splits.csv")
#   d <- live$score_plus_skills
#   role <- splits$role[match(d$match_id, splits$match_id)]
#   fold <- splits$fold[match(d$match_id, splits$match_id)]
#   is_test <- role == "test"

library(dplyr)

live_file <- "Data/derived/live_analysis.rds"
# The live wrangling input has no dates; this export has one per match.
dates_file <- "Data/conference_matches.csv"

live <- readRDS(live_file)
model_matches <- live$score_plus_skills |>
  distinct(match_id, opponent = visiting_team)

match_dates <- read.csv(dates_file) |>
  distinct(match_id, date) |>
  mutate(date = as.Date(date))
stopifnot(!anyDuplicated(match_dates$match_id),
          all(model_matches$match_id %in% match_dates$match_id))

# Seasons by date. The 2020 season was played in spring 2021 ("2021S").
splits <- model_matches |>
  left_join(match_dates, by = "match_id") |>
  mutate(
    season = case_when(
      date < as.Date("2019-01-01") ~ "2018",
      date < as.Date("2020-01-01") ~ "2019",
      date < as.Date("2021-06-01") ~ "2021S",
      date < as.Date("2022-01-01") ~ "2021",
      date < as.Date("2023-01-01") ~ "2022",
      date < as.Date("2024-01-01") ~ "2023",
      TRUE ~ "2024"
    ),
    conference = if_else(season == "2024", "Big 12", "Pac-12"),
    role = if_else(season %in% c("2023", "2024"), "test", "train"),
    fold = case_when(
      season %in% c("2018", "2019") ~ 1L,
      season == "2021S" ~ 2L,
      season == "2021" ~ 3L,
      season == "2022" ~ 4L,
      TRUE ~ NA_integer_
    )
  ) |>
  arrange(date, match_id) |>
  select(match_id, date, season, conference, opponent, role, fold)

stopifnot(nrow(splits) == length(unique(live$score_plus_skills$match_id)),
          all(!is.na(splits$fold) == (splits$role == "train")))

write.csv(splits, "model_splits.csv", row.names = FALSE)

# Summary so readers can check the split sizes.
live$score_plus_skills |>
  distinct(match_id, set_number) |>
  left_join(splits, by = "match_id") |>
  group_by(role, fold) |>
  summarise(seasons = paste(unique(season), collapse = ", "),
            matches = n_distinct(match_id), sets = n(), .groups = "drop") |>
  print()
