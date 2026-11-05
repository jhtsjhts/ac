from common import (
    convert_numeric_columns,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_teams():
    teams = load_csv("Teams.csv")

    teams = convert_numeric_columns(
        teams,
        [
            "year", "G", "W", "Pts", "GF", "GA",
        ],
    )

    games = teams["G"]

    teams["points_per_game"] = safe_divide(teams["Pts"], games)
    teams["goals_for_per_game"] = safe_divide(teams["GF"], games)
    teams["goals_against_per_game"] = safe_divide(teams["GA"], games)
    teams["goal_differential"] = teams["GF"] - teams["GA"]
    teams["goal_differential_per_game"] = safe_divide(
        teams["goal_differential"],
        games,
    )
    teams["win_rate"] = safe_divide(teams["W"], games)
    teams = teams[
        [
            "year",
            "lgID",
            "tmID",
            "rank",
            "G",
            "W",
            "Pts",
            "GF",
            "GA",
            "points_per_game",
            "win_rate",
            "goals_for_per_game",
            "goals_against_per_game",
            "goal_differential",
            "goal_differential_per_game",
        ]
    ]
    save_normalized(teams, "teams.csv")


if __name__ == "__main__":
    normalize_teams()