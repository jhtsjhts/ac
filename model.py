"""Simple baseline comparisons for ranking and coaching-change predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
    
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "normalized_data"
TEAM_FEATURES = [
    "points_per_game",
    "goal_differential_per_game",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized team and coach data."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    coaches = pd.read_csv(DATA_DIR / "coaches.csv")
    return teams, coaches


def available_history(years: list[int], target_year: int) -> list[int]:
    """Return up to three seasons available before the target season."""
    return sorted(
        [year for year in years if year < target_year],
        reverse=True,
    )[:3]


def season_rank(teams: pd.DataFrame) -> pd.DataFrame:
    """Create a consistent ranking using points and goal statistics."""
    teams = teams.copy()
    teams["goal_differential"] = teams["GF"] - teams["GA"]
    return (
        teams.sort_values(
            ["Pts", "goal_differential", "GF"],
            ascending=[False, False, False],
        )
        .assign(rank=lambda df: range(1, len(df) + 1))
    )


def weighted_team_score(
    teams: pd.DataFrame,
    team_id: str,
    history: list[int],
) -> float | None:
    """Calculate a recent-performance score for one team."""
    weights = [0.6, 0.3, 0.1]
    rows = teams[
        (teams["tmID"] == team_id)
        & teams["year"].isin(history)
    ].set_index("year")

    values = []
    used_weights = []
    for year, weight in zip(history, weights):
        if year not in rows.index:
            continue
        row = rows.loc[year]
        if row[TEAM_FEATURES].isna().any():
            continue
        values.append(
            0.7 * row["points_per_game"]
            + 0.3 * row["goal_differential_per_game"]
        )
        used_weights.append(weight)

    if not values:
        return None
    return sum(value * weight for value, weight in zip(values, used_weights)) / sum(
        used_weights
    )


def coach_at_season(
    coaches: pd.DataFrame,
    team_id: str,
    year: int,
) -> tuple[str, str] | None:
    """Return the first and final coach recorded for one team-season."""
    rows = coaches[
        (coaches["tmID"] == team_id)
        & (coaches["year"] == year)
    ].sort_values("stint")
    if rows.empty:
        return None
    return rows.iloc[0]["coachID"], rows.iloc[-1]["coachID"]


def build_season_predictions(
    teams: pd.DataFrame,
    coaches: pd.DataFrame,
    league: str,
    target_year: int,
) -> pd.DataFrame:
    """Build baseline and simple predictions for one historical season."""
    teams = teams[teams["lgID"] == league].copy()
    coaches = coaches[coaches["lgID"] == league].copy()
    years = sorted(teams["year"].dropna().astype(int).unique())
    history = available_history(years, target_year)
    if not history:
        return pd.DataFrame()

    previous_year = history[0]
    previous = season_rank(teams[teams["year"] == previous_year])
    actual = season_rank(teams[teams["year"] == target_year])
    current = teams[teams["year"] == target_year]
    previous_median = previous["points_per_game"].median()
    rows = []

    for team_id in current["tmID"]:
        previous_team = previous[previous["tmID"] == team_id]
        actual_team = actual[actual["tmID"] == team_id]
        score = weighted_team_score(teams, team_id, history)
        previous_coach = coach_at_season(coaches, team_id, previous_year)
        current_coach = coach_at_season(coaches, team_id, target_year)

        if (
            previous_team.empty
            or actual_team.empty
            or score is None
            or previous_coach is None
            or current_coach is None
        ):
            continue

        rows.append(
            {
                "tmID": team_id,
                "actual_rank": actual_team.iloc[0]["rank"],
                "baseline_rank": previous_team.iloc[0]["rank"],
                "simple_score": score,
                "actual_coach_change": int(
                    previous_coach[1] != current_coach[0]
                ),
                "baseline_coach_change": 0,
                "simple_coach_change": int(
                    previous_team.iloc[0]["points_per_game"]
                    < previous_median
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["simple_rank"] = (
        result["simple_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return result


def rank_metrics(
    predictions: pd.DataFrame,
    column: str,
) -> tuple[float, float]:
    """Return mean absolute rank error and Spearman correlation."""
    error = (
        predictions["actual_rank"] - predictions[column]
    ).abs().mean()
    correlation = predictions["actual_rank"].corr(
        predictions[column],
        method="spearman",
    )
    return error, correlation


def classification_metrics(
    predictions: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for coaching changes."""
    actual = predictions["actual_coach_change"]
    predicted = predictions[column]
    true_positive = ((actual == 1) & (predicted == 1)).sum()
    false_positive = ((actual == 0) & (predicted == 1)).sum()
    false_negative = ((actual == 1) & (predicted == 0)).sum()
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive
    ) else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative
    ) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def report(predictions: pd.DataFrame, year: int) -> None:
    """Print only predictions and the minimum baseline comparison."""
    print(f"\nSeason {year}")
    print("\nRanking")
    print(
        predictions[
            ["tmID", "actual_rank", "simple_rank"]
        ]
        .sort_values("actual_rank")
        .to_string(index=False)
    )

    actual_changes = sorted(
        predictions.loc[
            predictions["actual_coach_change"] == 1,
            "tmID",
        ].tolist()
    )
    predicted_changes = sorted(
        predictions.loc[
            predictions["simple_coach_change"] == 1,
            "tmID",
        ].tolist()
    )
    print("\nCoaching changes")
    print(f"Actual:    {actual_changes}")
    print(f"Predicted: {predicted_changes}")

    print("\nBaseline comparison")
    baseline_mae, _ = rank_metrics(
        predictions,
        "baseline_rank",
    )
    simple_mae, _ = rank_metrics(
        predictions,
        "simple_rank",
    )
    print(
        f"Average ranking error: baseline={baseline_mae:.3f}, "
        f"prediction={simple_mae:.3f}"
    )

    _, _, baseline_f1 = classification_metrics(
        predictions,
        "baseline_coach_change",
    )
    _, _, simple_f1 = classification_metrics(
        predictions,
        "simple_coach_change",
    )
    print(
        f"Coaching prediction F1: baseline={baseline_f1:.3f}, "
        f"prediction={simple_f1:.3f}"
    )


def main() -> None:
    """Evaluate one season or all evaluable seasons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--league", default="NHL")
    args = parser.parse_args()

    teams, coaches = load_data()
    years = sorted(
        teams.loc[teams["lgID"] == args.league, "year"]
        .dropna()
        .astype(int)
        .unique()
    )
    target_years = [args.year] if args.year else years
    results = []

    for year in target_years:
        predictions = build_season_predictions(
            teams,
            coaches,
            args.league,
            year,
        )
        if predictions.empty:
            continue
        report(predictions, year)
        results.append(predictions.assign(target_year=year))

    if not results:
        raise ValueError("No season had enough previous data for evaluation.")

    if args.year is None:
        all_results = pd.concat(results, ignore_index=True)
        print("\nOverall results")
        report(all_results, "all")


if __name__ == "__main__":
    main()
