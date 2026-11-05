# Normalization and Model Explanation

## 1. Purpose of the project

The project uses historical hockey data to answer two questions:

1. How can teams, players, goalies, and coaches be compared fairly across seasons?
2. Can recent team performance provide a useful prediction of the next season's ranking or coaching changes?

The raw data contains totals from seasons with very different schedules, league sizes, rules, and scoring environments. The normalization scripts transform those totals into comparable variables. The model then uses a small selection of the normalized team variables to create simple historical predictions.

The implementation described here is in [`normalization_scripts/`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/normalization_scripts>) and [`model.py`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/model.py>).

---

## 2. Data normalization

### 2.1 What normalization means here

In this project, normalization does not mean splitting the data into database tables under 1NF, 2NF, or 3NF. It means cleaning the CSV data and deriving rates that make historical records more comparable.

For example, comparing a team's total goals in a 22-game season with a team's total goals in an 82-game season would be misleading. Dividing by games played produces a per-game statistic that is easier to compare.

The scripts:

1. Read files from `raw_data/`.
2. Interpret empty values and common missing-value strings as missing data.
3. Convert selected columns to numeric types.
4. Calculate rates and derived statistics.
5. Write new files to `normalized_data/`.

The raw files are not modified.

### 2.2 Shared utility functions

The common functions are defined in [`normalization_scripts/common.py`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/normalization_scripts/common.py>).

#### Loading CSV files

`load_csv()` reads a CSV from the `raw_data` directory. Empty strings, spaces, `NA`, and `N/A` are converted to pandas missing values (`NaN`).

#### Numeric conversion

`convert_numeric_columns()` converts specified columns with `pandas.to_numeric()`. Invalid numeric text is converted to missing data rather than causing the whole process to fail.

#### Safe division

`safe_divide()` replaces zero denominators with missing values before dividing. Therefore, a statistic such as `wins / games` is missing when the number of games is zero instead of producing an infinite value.

#### Saving output

`save_normalized()` creates the `normalized_data` directory if necessary and writes the result as a CSV without including the pandas index.

---

## 3. Team normalization

The team script is [`normalization_scripts/normalize_teams.py`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/normalization_scripts/normalize_teams.py>). It reads `raw_data/Teams.csv` and writes `normalized_data/teams.csv`.

### 3.1 Basic team statistics

| Statistic | Formula | Interpretation |
|---|---|---|
| `points_per_game` | `Pts / G` | Average standings points obtained per game |
| `win_rate` | `W / G` | Proportion of games won |
| `goals_for_per_game` | `GF / G` | Average goals scored per game |
| `goals_against_per_game` | `GA / G` | Average goals conceded per game |
| `goal_differential` | `GF - GA` | Total goals scored minus goals conceded |
| `goal_differential_per_game` | `(GF - GA) / G` | Average scoring margin per game |

The original identifiers and totals are retained together with these derived values:

```text
year, lgID, tmID, rank, G, W, Pts, GF, GA
```

The definitions are also listed in [`normalized_glossary.md`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/normalized_glossary.md>).

### 3.2 Why these statistics are useful

- `points_per_game` adjusts points for different season lengths.
- `win_rate` adjusts wins for different numbers of games.
- `goals_for_per_game` measures offensive production.
- `goals_against_per_game` measures defensive performance.
- `goal_differential_per_game` combines offense and defense into one scoring-margin measure.

The model uses `points_per_game` and `goal_differential_per_game` because they provide a compact description of team quality.

---

## 4. Coach normalization

The coach script is [`normalization_scripts/normalize_coaches.py`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/normalization_scripts/normalize_coaches.py>). It reads `raw_data/Coaches.csv` and writes `normalized_data/coaches.csv`.

It converts the season, stint, game, win, loss, tie, and postseason columns to numeric values and calculates:

| Statistic | Formula | Interpretation |
|---|---|---|
| `win_rate` | `w / g` | Regular-season wins per game coached |
| `postseason_win_rate` | `postw / postg` | Postseason wins per postseason game coached |

The coach output also retains:

```text
coachID, year, tmID, lgID, stint, g, w, postg, postw
```

The coach statistics are not used to calculate the team ranking in the current model. They are used to identify whether the coach changed between two seasons.

---

## 5. Historical context

The current model in `ac/model.py` uses the simpler normalized files from `ac/`. The more extensive historical-context functions are implemented in the separate [`ac_2/normalization_scripts/`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac_2/normalization_scripts>) version.

That newer pipeline adds:

- Era labels such as `early_league`, `expansion`, `dead_puck`, and `post_lockout`
- Indicators for ties, overtime losses, and shootouts
- The expected number of NHL games in each season
- The number of teams qualifying for the playoffs
- Within-season and within-league z-scores
- Normalized player and goalie statistics

These features are useful because the league changed substantially over time. For example, schedules ranged from fewer than 20 games to 82 games, and the number of playoff teams also changed. The historical mappings are documented in [`changes.md`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/changes.md>).

---

## 6. What the model does

The implementation is a **simple baseline comparison and historical backtest**, not a fitted machine-learning model. It does not train a regression, classification, or neural-network model.

The model is implemented in [`model.py`](</home/koikoikoikoi/Documents/uni/meic/1ºs/ac/ac/model.py>).

It evaluates each target season using only seasons that occurred before it. This avoids using future information when constructing a prediction.

### 6.1 Input data

`load_data()` loads:

```text
normalized_data/teams.csv
normalized_data/coaches.csv
```

The default league is `NHL`, although another league can be selected with `--league`.

### 6.2 Historical window

For a target season, `available_history()` selects up to the three most recent seasons before that target year.

For example, if the target season is 2009 and the available previous seasons are 2008, 2007, and 2006, the history is ordered as:

```text
[2008, 2007, 2006]
```

The intended weights are:

```text
most recent season: 0.6
second most recent: 0.3
third most recent: 0.1
```

If a team is missing one of those seasons, the available weights are renormalized so that they still sum to one.

### 6.3 Actual and baseline rankings

`season_rank()` creates a consistent ranking by sorting teams in this order:

1. `Pts`, descending
2. `GF - GA`, descending
3. `GF`, descending

The resulting sequential positions are assigned as ranks from 1 onward.

The model calculates:

- `actual_rank`: ranking created from the target season
- `baseline_rank`: ranking from the immediately preceding season
- `simple_rank`: ranking from the model's weighted recent-performance score

This ranking is deliberately reconstructed from points and goal statistics rather than relying directly on the historical `rank` column. The historical meaning of `rank` changed across eras, so it is not always comparable between seasons.

### 6.4 Weighted team score

For each team and each available historical season, the model calculates:

```text
season_score =
    0.7 * points_per_game
  + 0.3 * goal_differential_per_game
```

The season scores are then combined using the recency weights:

```text
simple_score =
    0.6 * most_recent_score
  + 0.3 * second_recent_score
  + 0.1 * third_recent_score
```

The team with the largest `simple_score` receives `simple_rank = 1`.

The score gives more importance to points per game than to goal differential per game:

- 70% of each season's score comes from points per game.
- 30% comes from goal differential per game.

The model therefore assumes that recent results are more informative than older results and that standings performance is more important than scoring margin.

---

## 7. Coaching-change prediction

The model compares the first and final coach recorded for each team-season using `coach_at_season()`.

### 7.1 Actual coaching change

The actual label is calculated as:

```text
actual_coach_change = 1
```

when the final coach of the previous season is different from the first coach of the target season.

Otherwise:

```text
actual_coach_change = 0
```

This represents whether a team began the target season with a different coach from the one who finished the previous season.

### 7.2 Baseline prediction

The baseline always predicts no coaching change:

```text
baseline_coach_change = 0
```

This is a deliberately simple reference point. A more sophisticated rule must perform better than this baseline to be useful.

### 7.3 Simple prediction rule

The simple rule predicts a coaching change when the team's previous-season `points_per_game` is below the median points per game of all teams in that previous season:

```text
simple_coach_change =
    1 if previous_team_points_per_game < previous_season_median
    0 otherwise
```

This tests the hypothesis that teams performing below the previous season's middle point are more likely to change coaches.

It is a heuristic, not a causal claim. Poor performance may be associated with coaching changes, but this model does not prove that poor performance causes them.

---

## 8. Statistics printed by the model

For each evaluable season, `report()` prints four sections.

### 8.1 Ranking table

The ranking table contains:

| Column | Meaning |
|---|---|
| `tmID` | Team identifier |
| `actual_rank` | Rank reconstructed from the target season's results |
| `baseline_rank` | Previous-season rank |
| `simple_rank` | Rank produced by the weighted recent-performance score |

The table is sorted by `actual_rank`, so the strongest actual team appears first.

### 8.2 Coaching-change lists

The model prints:

- `Actual`: teams for which a coaching change really occurred
- `Predicted`: teams selected by the simple below-median rule
- `Baseline`: always an empty list because the baseline predicts no changes

These lists show exactly which teams are counted as positive cases for the coaching-change metrics.

### 8.3 Ranking MAE

`Ranking MAE` means **mean absolute error** between predicted and actual ranks:

```text
MAE = mean(abs(actual_rank - predicted_rank))
```

Interpretation:

- `0` means perfect rank predictions.
- A value of `2` means predictions are, on average, two ranking positions away from the actual rank.
- Lower is better.

The model reports MAE for:

- `baseline_rank`
- `simple_rank`

### 8.4 Spearman rank correlation

`Ranking Spearman` is Spearman's rank correlation between predicted and actual rankings.

Interpretation:

- `1` means the order is identical.
- `0` means there is no monotonic relationship.
- `-1` means the order is reversed.
- Higher is better.

This metric evaluates whether the model identifies the ordering of teams, even when individual positions are not exact.

### 8.5 Precision

Precision answers:

> Of the teams predicted to change coaches, how many actually changed coaches?

```text
precision = true positives / (true positives + false positives)
```

High precision means the model's coaching-change alerts are usually correct.

### 8.6 Recall

Recall answers:

> Of all teams that actually changed coaches, how many did the model identify?

```text
recall = true positives / (true positives + false negatives)
```

High recall means the model misses relatively few actual changes.

### 8.7 F1 score

The F1 score combines precision and recall using their harmonic mean:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Interpretation:

- `1` means perfect precision and recall.
- `0` means the model has no useful positive predictions.
- Higher is better.

F1 is useful here because coaching changes may be less frequent than non-changes. Accuracy alone could look high simply by predicting no change every time.

---

## 9. Overall results

When the program is run without `--year`, it evaluates every season for which enough previous data exists.

It then concatenates all per-season prediction tables and prints an `Overall results` section. The same ranking and coaching metrics are recalculated over all evaluated team-season rows.

This gives an aggregate view, but it should be interpreted carefully:

- Each season can have a different number of teams.
- Historical rules and schedules changed.
- Rows from different seasons are not completely independent.
- The model is intentionally simple and should be treated as a baseline for comparison.

## 10. Running the model

Evaluate every possible NHL season:

```bash
python model.py
```

Evaluate one season:

```bash
python model.py --year 2009
```

Evaluate another league:

```bash
python model.py --league WNBA
```

The selected league must exist in the normalized team and coach CSV files.

## 11. Main limitations

The current implementation is useful as a transparent baseline, but it has several limitations:

1. It uses only two team features: points per game and goal differential per game.
2. It does not train model parameters from data.
3. The coaching-change rule uses only the previous season's points per game.
4. It does not account for injuries, payroll, player quality, ownership, or other contextual factors.
5. The historical `rank` field is not used directly because its meaning changes across eras.
6. Missing seasons or missing team records can cause a team to be excluded from a prediction table.
7. Historical changes in league rules and schedules can still affect comparability, even after per-game normalization.

The main value of this implementation is therefore interpretability: every prediction and every metric can be traced directly to a small number of explicit formulas.
