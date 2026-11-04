from common import (
    add_season_z_scores,
    convert_numeric_columns,
    keep_model_window,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_goalies():
    goalies = keep_model_window(load_csv("Goalies.csv"))

    goalies = convert_numeric_columns(
        goalies,
        [
            "year", "stint", "GP", "Min", "W", "L",
            "T/OL", "ENG", "SHO", "GA", "SA",
        ],
    )

    goalies["wins_per_game"] = safe_divide(
        goalies["W"],
        goalies["GP"],
    )

    goalies["goals_against_per_game"] = safe_divide(
        goalies["GA"],
        goalies["GP"],
    )

    goalies["goals_against_per_60_minutes"] = safe_divide(
        goalies["GA"] * 60,
        goalies["Min"],
    )

    goalies["shutouts_per_game"] = safe_divide(
        goalies["SHO"],
        goalies["GP"],
    )

    goalies["save_percentage"] = 1 - safe_divide(
        goalies["GA"],
        goalies["SA"],
    )

    goalies = add_season_z_scores(
        goalies,
        [
            "wins_per_game",
            "goals_against_per_game",
            "goals_against_per_60_minutes",
            "shutouts_per_game",
            "save_percentage",
        ],
    )

    save_normalized(goalies, "goalies.csv")


if __name__ == "__main__":
    normalize_goalies()