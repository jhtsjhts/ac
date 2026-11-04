from common import (
    add_season_z_scores,
    convert_numeric_columns,
    keep_model_window,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_players():
    scoring = keep_model_window(load_csv("Scoring.csv"))

    scoring = convert_numeric_columns(
        scoring,
        [
            "year", "stint", "GP", "G", "A", "Pts", "PIM",
            "+/-", "PPG", "PPA", "SHG", "SHA", "GWG",
            "GTG", "SOG",
        ],
    )

    games = scoring["GP"]

    scoring["goals_per_game"] = safe_divide(scoring["G"], games)
    scoring["assists_per_game"] = safe_divide(scoring["A"], games)
    scoring["points_per_game"] = safe_divide(scoring["Pts"], games)

    scoring["penalty_minutes_per_game"] = safe_divide(
        scoring["PIM"],
        games,
    )

    scoring["power_play_goals_per_game"] = safe_divide(
        scoring["PPG"],
        games,
    )

    scoring["power_play_assists_per_game"] = safe_divide(
        scoring["PPA"],
        games,
    )

    scoring["short_handed_goals_per_game"] = safe_divide(
        scoring["SHG"],
        games,
    )

    scoring["game_winning_goals_per_game"] = safe_divide(
        scoring["GWG"],
        games,
    )

    scoring["shooting_percentage"] = safe_divide(
        scoring["G"],
        scoring["SOG"],
    )

    scoring = add_season_z_scores(
        scoring,
        [
            "points_per_game",
            "goals_per_game",
            "assists_per_game",
            "shooting_percentage",
        ],
    )

    save_normalized(scoring, "scoring.csv")


if __name__ == "__main__":
    normalize_players()