"""Predict in-season coaching changes using only information from earlier seasons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
IDENTIFIERS = ["year", "lgID", "tmID", "franchID", "name"]
FEATURES = [
    "previous_points_per_game",
    "previous_win_rate",
    "previous_goals_for_per_game",
    "previous_goals_against_per_game",
    "previous_goal_difference_per_game",
    "previous_points_vs_median",
    "previous_made_playoffs",
    "previous_coach_change",
    "previous_coach_count",
    "previous_final_coach_win_rate",
    "points_per_game_trend",
    "goal_difference_trend",
    "mean_points_per_game_last3",
    "mean_coach_changes_last3",
    "previous_final_coach_team_tenure",
    "previous_final_coach_career_seasons",
    "prior_team_seasons",
    "has_previous_league_season",
    "years_since_last_appearance",
]
PREVIOUS_STATS = [
    "points_per_game", "win_rate", "goals_for_per_game",
    "goals_against_per_game", "goal_difference_per_game",
    "points_vs_median", "made_playoffs", "coach_change", "coach_count",
    "final_coach_win_rate",
]


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns: {', '.join(missing)}")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw tables; no normalization script needs to run beforehand."""
    return pd.read_csv(data_dir / "Teams.csv"), pd.read_csv(data_dir / "Coaches.csv")


def prepare_history(
    teams: pd.DataFrame, coaches: pd.DataFrame, league: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize team-seasons, preserving missing coach labels as unknown."""
    require_columns(teams, IDENTIFIERS + ["G", "W", "Pts", "GF", "GA", "playoff"], "Teams")
    require_columns(coaches, ["year", "lgID", "tmID", "coachID", "stint", "g", "w"], "Coaches")
    teams = teams.loc[teams["lgID"].eq(league)].copy()
    coaches = coaches.loc[coaches["lgID"].eq(league)].copy()
    if teams.empty:
        raise ValueError(f"No teams found for league {league!r}.")
    for column in ["year", "G", "W", "Pts", "GF", "GA"]:
        teams[column] = pd.to_numeric(teams[column], errors="raise")
    for column in ["year", "stint", "g", "w"]:
        coaches[column] = pd.to_numeric(coaches[column], errors="coerce")
    teams["year"] = teams["year"].astype(int)
    teams["franchID"] = teams["franchID"].fillna(teams["tmID"])
    if teams.duplicated(["year", "tmID"]).any() or teams.duplicated(["year", "franchID"]).any():
        raise ValueError("Teams must have one row per team/franchise and season in the selected league.")
    games = teams["G"].replace(0, np.nan)
    teams["points_per_game"] = teams["Pts"] / games
    teams["win_rate"] = teams["W"] / games
    teams["goals_for_per_game"] = teams["GF"] / games
    teams["goals_against_per_game"] = teams["GA"] / games
    teams["goal_difference_per_game"] = (teams["GF"] - teams["GA"]) / games
    teams["points_vs_median"] = teams["points_per_game"] - teams.groupby("year")["points_per_game"].transform("median")
    teams["made_playoffs"] = teams["playoff"].fillna("").astype(str).str.strip().ne("").astype(int)

    # Missing game totals can still identify a documented stint (e.g. a returning
    # coach). Explicit zero-game entries do not establish an in-season change.
    coaches = coaches.loc[coaches["g"].isna() | coaches["g"].gt(0)].copy()
    summaries = []
    for (year, team_id), group in coaches.groupby(["year", "tmID"]):
        valid = group.dropna(subset=["coachID", "stint"]).sort_values(["stint", "coachID"])
        if valid.empty:
            continue
        final = valid.loc[valid["stint"].eq(valid["stint"].max())]
        games_coached = final["g"].sum(min_count=1)
        final_win_rate = final["w"].sum(min_count=1) / games_coached if games_coached > 0 else np.nan
        complete = len(valid) == len(group)
        summaries.append({
            "year": int(year), "tmID": team_id,
            # Co-coaches sharing one stint are not a coaching replacement.
            "coach_change": int(valid["stint"].nunique() > 1 and valid["coachID"].nunique() > 1) if complete else np.nan,
            "coach_count": valid["coachID"].nunique(),
            "final_coachID": final.iloc[0]["coachID"],
            "final_coach_win_rate": final_win_rate,
        })
    summary = pd.DataFrame(summaries, columns=[
        "year", "tmID", "coach_change", "coach_count", "final_coachID", "final_coach_win_rate",
    ])
    seasons = teams.merge(summary, on=["year", "tmID"], how="left", validate="one_to_one")
    return seasons.sort_values(["year", "tmID"]).reset_index(drop=True), coaches


def build_dataset(
    seasons: pd.DataFrame, coaches: pd.DataFrame, roster: pd.DataFrame,
) -> pd.DataFrame:
    """Build features with a strict year < target_year filter for every row.

    Only identifiers come from the target roster. Target-season outcomes are
    joined as labels after feature construction, never as model inputs.
    """
    require_columns(roster, IDENTIFIERS, "Roster")
    if roster.duplicated(["year", "lgID", "tmID"]).any() or roster.duplicated(["year", "lgID", "franchID"]).any():
        raise ValueError("Roster contains duplicate teams/franchises within a season.")
    labels = seasons.set_index(["year", "tmID"])["coach_change"]
    histories = {key: group for key, group in seasons.groupby("franchID")}
    league_years = sorted(seasons["year"].unique())
    rows = []
    for team in roster[IDENTIFIERS].to_dict("records"):
        year = int(team["year"])
        history = histories.get(team["franchID"], seasons.iloc[:0])
        history = history.loc[history["year"].lt(year)].sort_values("year")
        previous_league_years = [value for value in league_years if value < year]
        row = {**team, **dict.fromkeys(FEATURES, np.nan)}
        row["feature_season"] = np.nan
        row["prior_team_seasons"] = len(history)
        row["has_previous_league_season"] = 0
        if not history.empty:
            previous = history.iloc[-1]
            row["feature_season"] = int(previous["year"])
            row["has_previous_league_season"] = int(previous["year"] == previous_league_years[-1])
            row["years_since_last_appearance"] = year - int(previous["year"])
            for stat in PREVIOUS_STATS:
                row[f"previous_{stat}"] = previous[stat]
            row["mean_points_per_game_last3"] = history.tail(3)["points_per_game"].mean()
            row["mean_coach_changes_last3"] = history.tail(3)["coach_change"].mean()
            if len(history) >= 2:
                older = history.iloc[-2]
                row["points_per_game_trend"] = previous["points_per_game"] - older["points_per_game"]
                row["goal_difference_trend"] = previous["goal_difference_per_game"] - older["goal_difference_per_game"]
            coach_id = previous["final_coachID"]
            if pd.notna(coach_id):
                tenure = 0
                for value in history["final_coachID"].iloc[::-1]:
                    if pd.isna(value) or value != coach_id:
                        break
                    tenure += 1
                row["previous_final_coach_team_tenure"] = tenure
                row["previous_final_coach_career_seasons"] = coaches.loc[
                    coaches["year"].lt(year) & coaches["coachID"].eq(coach_id), "year",
                ].nunique()
        row["actual_coach_change"] = labels.get((year, team["tmID"]), np.nan)
        rows.append(row)
    if not rows:
        raise ValueError("Roster is empty.")
    return pd.DataFrame(rows).sort_values(["year", "tmID"]).reset_index(drop=True)


def make_model(regularization: float) -> Pipeline:
    """Learn imputation, scaling and classifier parameters on training rows only."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=regularization, class_weight="balanced", max_iter=3000, random_state=42,
        )),
    ])


def select_parameters(
    past: pd.DataFrame, validation_seasons: int,
) -> tuple[float, float, pd.DataFrame, dict]:
    """Choose C and the probability threshold on a chronological validation block."""
    years = sorted(past["year"].unique())
    if validation_seasons < 1 or len(years) < validation_seasons + 5:
        raise ValueError("Need at least five training seasons plus the requested validation seasons.")
    validation_years = years[-validation_seasons:]
    train = past.loc[past["year"].lt(validation_years[0])]
    validation = past.loc[past["year"].isin(validation_years)]
    if train["actual_coach_change"].nunique() < 2 or validation["actual_coach_change"].nunique() < 2:
        raise ValueError("Training and validation each need examples of both classes; choose a later year or a different validation window.")
    candidates = []
    for regularization in [0.01, 0.1, 1.0, 10.0]:
        model = make_model(regularization)
        model.fit(train[FEATURES], train["actual_coach_change"].astype(int))
        probabilities = model.predict_proba(validation[FEATURES])[:, 1]
        actual = validation["actual_coach_change"].astype(int)
        thresholds = []
        for threshold in np.linspace(0.1, 0.9, 33):
            predicted = probabilities >= threshold
            thresholds.append({
                "C": regularization, "threshold": float(threshold),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "average_precision": float(average_precision_score(actual, probabilities)),
            })
        candidates.append(max(thresholds, key=lambda item: (
            item["f1"], item["precision"], -abs(item["threshold"] - 0.5),
        )))
    selected = max(candidates, key=lambda item: (
        item["f1"], item["average_precision"], item["precision"], -item["C"],
    ))
    split = {
        "selection_training_years": [int(value) for value in years if value < validation_years[0]],
        "validation_years": [int(value) for value in validation_years],
        "selection_training_rows": len(train), "validation_rows": len(validation),
    }
    return selected["C"], selected["threshold"], pd.DataFrame(candidates), split


def classification_metrics(actual, predicted, probabilities=None) -> dict:
    """Evaluate positive predictions; handle single-class test seasons explicitly."""
    result = {
        "teams": len(actual), "actual_changes": int(np.sum(actual)),
        "predicted_changes": int(np.sum(predicted)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)) if len(np.unique(actual)) == 2 else None,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }
    if probabilities is not None:
        result["average_precision"] = float(average_precision_score(actual, probabilities)) if np.sum(actual) > 0 else None
        result["roc_auc"] = float(roc_auc_score(actual, probabilities)) if len(np.unique(actual)) == 2 else None
        result["positive_prevalence"] = float(np.mean(actual))
    return result


def train_and_evaluate(dataset: pd.DataFrame, target_year: int, validation_seasons: int = 5):
    past = dataset.loc[dataset["year"].lt(target_year) & dataset["actual_coach_change"].notna()].copy()
    target = dataset.loc[dataset["year"].eq(target_year)].copy()
    if target.empty:
        raise ValueError(f"No roster found for target season {target_year}.")
    regularization, threshold, scores, split = select_parameters(past, validation_seasons)
    # After model selection, refit on all labeled seasons preceding the test.
    model = make_model(regularization)
    model.fit(past[FEATURES], past["actual_coach_change"].astype(int))
    predictions = target[IDENTIFIERS + ["feature_season", "actual_coach_change"]].copy()
    predictions["actual_coach_change"] = predictions["actual_coach_change"].astype("Int64")
    predictions["change_probability"] = model.predict_proba(target[FEATURES])[:, 1]
    predictions["predicted_coach_change"] = predictions["change_probability"].ge(threshold).astype(int)
    predictions["baseline_no_change"] = 0
    predictions["baseline_all_change"] = 1
    predictions["baseline_below_median"] = target["previous_points_vs_median"].lt(0).astype(int)
    metrics = {
        "target_year": target_year, "league": str(target.iloc[0]["lgID"]),
        "label_definition": "Multiple distinct coaching stints and coaches within the target season; not an offseason replacement.",
        "model": "LogisticRegression", "C": regularization, "threshold": threshold,
        "selection": split, "final_training_years": sorted(past["year"].astype(int).unique().tolist()),
        "final_training_rows": len(past),
        "final_training_positive_prevalence": float(past["actual_coach_change"].mean()),
        "target_teams": len(target), "target_teams_without_labels": int(target["actual_coach_change"].isna().sum()),
        "historical_rows_without_labels": int((dataset["year"].lt(target_year) & dataset["actual_coach_change"].isna()).sum()),
        "test": None,
    }
    known = predictions.loc[predictions["actual_coach_change"].notna()]
    if not known.empty:
        actual = known["actual_coach_change"].to_numpy(dtype=int)
        metrics["test"] = {
            "model": classification_metrics(actual, known["predicted_coach_change"], known["change_probability"]),
            **{
                column: classification_metrics(actual, known[column])
                for column in ["baseline_no_change", "baseline_all_change", "baseline_below_median"]
            },
        }
    bundle = {
        "pipeline": model, "features": FEATURES, "threshold": threshold,
        "target_year": target_year, "league": metrics["league"],
        "training_last_year": int(past["year"].max()),
        "label_definition": metrics["label_definition"],
    }
    return bundle, predictions.sort_values("change_probability", ascending=False), metrics, scores


def save_results(output_dir: Path, bundle, predictions, metrics, scores, dataset) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    dataset.to_csv(output_dir / "dataset.csv", index=False)
    scores.to_csv(output_dir / "validation_scores.csv", index=False)
    predictions[IDENTIFIERS[1:]].sort_values("tmID").to_csv(output_dir / "roster_template.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(bundle, output_dir / "model.joblib")
    pipeline = bundle["pipeline"]
    coefficients = pd.DataFrame({
        "feature": pipeline.named_steps["imputer"].get_feature_names_out(FEATURES),
        "standardized_coefficient": pipeline.named_steps["classifier"].coef_[0],
    })
    coefficients.sort_values("standardized_coefficient", key=abs, ascending=False).to_csv(output_dir / "coefficients.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="Target season's starting year; defaults to the latest season in Teams.csv.")
    parser.add_argument("--league", default="NHL")
    parser.add_argument("--validation-seasons", type=int, default=5)
    parser.add_argument("--roster", type=Path, help="Preseason roster CSV with tmID, franchID and name; required for an unavailable season.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        teams, coaches = load_data(ROOT / "raw_data")
        # Discard later records before any preprocessing or label construction.
        league_teams = teams.loc[teams["lgID"].eq(args.league)]
        if league_teams.empty:
            raise ValueError(f"No teams found for league {args.league!r}.")
        target_year = args.year if args.year is not None else int(league_teams["year"].max())
        teams = teams.loc[teams["year"].le(target_year)]
        coaches = coaches.loc[coaches["year"].le(target_year)]
        seasons, coaches = prepare_history(teams, coaches, args.league)
        roster = seasons[IDENTIFIERS].copy()
        if args.roster:
            supplied = pd.read_csv(args.roster)
            require_columns(supplied, ["tmID", "franchID", "name"], "Roster")
            if "lgID" in supplied:
                supplied = supplied.loc[supplied["lgID"].eq(args.league)].copy()
            supplied["year"] = target_year
            supplied["lgID"] = args.league
            if supplied.empty:
                raise ValueError("Supplied roster has no teams in the selected league.")
            if supplied[["tmID", "franchID", "name"]].isna().any().any():
                raise ValueError("Roster identifiers and team names cannot be missing.")
            roster = pd.concat([roster.loc[roster["year"].lt(target_year)], supplied[IDENTIFIERS]], ignore_index=True)
        elif not roster["year"].eq(target_year).any():
            raise ValueError("Target season is unavailable. Supply its known preseason teams using --roster path/to/roster.csv.")
        dataset = build_dataset(seasons, coaches, roster)
        bundle, predictions, metrics, scores = train_and_evaluate(dataset, target_year, args.validation_seasons)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output_dir = args.output_dir or ROOT / "outputs" / "coach_model" / args.league / str(target_year)
    save_results(output_dir, bundle, predictions, metrics, scores, dataset)
    split = metrics["selection"]
    print(f"League: {args.league} | target season: {target_year}")
    print(f"Selection training: {split['selection_training_years'][0]}-{split['selection_training_years'][-1]}")
    print(f"Validation seasons: {split['validation_years']}")
    print(f"Final training: {metrics['final_training_rows']} rows, all before {target_year}")
    print(f"Selected C={bundle['pipeline'].named_steps['classifier'].C:g}, threshold={bundle['threshold']:.3f}")
    print("\n" + predictions[["tmID", "name", "change_probability", "predicted_coach_change", "actual_coach_change"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if metrics["test"]:
        print("\nTest results (only teams with known labels)")
        for name, values in metrics["test"].items():
            print(f"{name}: precision={values['precision']:.3f}, recall={values['recall']:.3f}, F1={values['f1']:.3f}")
    else:
        print("\nPredictions only: target-season labels are unavailable.")
    changed = predictions.loc[predictions["predicted_coach_change"].eq(1), "tmID"].tolist()
    print(f"\nTeams predicted to change coaches: {changed}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
