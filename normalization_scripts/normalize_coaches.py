from common import (
    add_season_z_scores,
    convert_numeric_columns,
    keep_model_window,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_coaches():
    coaches = keep_model_window(load_csv("Coaches.csv"))

    coaches = convert_numeric_columns(
        coaches,
        [
            "year", "stint", "g", "w", "l", "t",
            "postg", "postw", "postl", "postt",
        ],
    )

    coaches["win_rate"] = safe_divide(
        coaches["w"],
        coaches["g"],
    )

    coaches["loss_rate"] = safe_divide(
        coaches["l"],
        coaches["g"],
    )

    coaches["tie_rate"] = safe_divide(
        coaches["t"],
        coaches["g"],
    )

    coaches["postseason_win_rate"] = safe_divide(
        coaches["postw"],
        coaches["postg"],
    )

    coaches["postseason_qualification"] = (
        coaches["postg"].fillna(0) > 0
    ).astype(int)

    coaches["coach_tenure_indicator"] = (
        coaches["stint"] == 1
    ).astype(int)

    coaches = add_season_z_scores(
        coaches,
        [
            "win_rate",
            "loss_rate",
            "postseason_win_rate",
        ],
    )

    save_normalized(coaches, "coaches.csv")


if __name__ == "__main__":
    normalize_coaches()