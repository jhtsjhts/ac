import numpy as np

from common import (
    add_season_z_scores,
    convert_numeric_columns,
    keep_model_window,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_teams():
    teams = keep_model_window(load_csv("Teams.csv"))

    teams = convert_numeric_columns(
        teams,
        [
            "year", "rank", "G", "W", "L", "T", "OTL", "Pts",
            "SoW", "SoL", "GF", "GA", "PIM", "BenchMinor",
            "PPG", "PPC", "SHA", "PKG", "PKC", "SHF",
        ],
    )

    games = teams["G"]

    teams["wins_per_game"] = safe_divide(teams["W"], games)
    teams["losses_per_game"] = safe_divide(teams["L"], games)
    teams["ties_per_game"] = safe_divide(teams["T"], games)
    teams["ot_losses_per_game"] = safe_divide(teams["OTL"], games)
    teams["shootout_wins_per_game"] = safe_divide(teams["SoW"], games)
    teams["shootout_losses_per_game"] = safe_divide(teams["SoL"], games)

    teams["points_per_game"] = safe_divide(teams["Pts"], games)
    teams["points_rate"] = safe_divide(teams["Pts"], 2 * games)

    teams["goals_for_per_game"] = safe_divide(teams["GF"], games)
    teams["goals_against_per_game"] = safe_divide(teams["GA"], games)

    teams["goal_differential"] = teams["GF"] - teams["GA"]
    teams["goal_differential_per_game"] = safe_divide(
        teams["goal_differential"],
        games,
    )

    teams["win_rate"] = safe_divide(teams["W"], games)

    teams["penalty_minutes_per_game"] = safe_divide(
        teams["PIM"],
        games,
    )

    teams["bench_minor_per_game"] = safe_divide(
        teams["BenchMinor"],
        games,
    )

    teams["power_play_conversion_rate"] = safe_divide(
        teams["PPG"],
        teams["PPC"],
    )

    teams["penalty_kill_rate"] = 1 - safe_divide(
        teams["SHA"],
        teams["PKC"],
    )

    teams["short_handed_goals_per_game"] = safe_divide(
        teams["SHF"],
        games,
    )

    teams = add_season_z_scores(
        teams,
        [
            "points_per_game",
            "points_rate",
            "goals_for_per_game",
            "goals_against_per_game",
            "goal_differential_per_game",
            "win_rate",
            "power_play_conversion_rate",
            "penalty_kill_rate",
            "penalty_minutes_per_game",
        ],
    )

    teams["era"] = np.select(
        [
            teams["year"].between(1979, 1994),
            teams["year"].between(1995, 2003),
            teams["year"].between(2005, 2009),
        ],
        [
            "pre_dead_puck",
            "dead_puck",
            "post_lockout",
        ],
        default="historical",
    )

    save_normalized(teams, "teams.csv")


if __name__ == "__main__":
    normalize_teams()