from common import (
    convert_numeric_columns,
    load_csv,
    safe_divide,
    save_normalized,
)


def normalize_coaches():
    coaches = load_csv("Coaches.csv")

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
    coaches["postseason_win_rate"] = safe_divide(
        coaches["postw"],
        coaches["postg"],
    )

    coaches = coaches[
        [
            "coachID",
            "year",
            "tmID",
            "lgID",
            "stint",
            "g",
            "w",
            "postg",
            "postw",
            "win_rate",
            "postseason_win_rate",
        ]
    ]
    save_normalized(coaches, "coaches.csv")


if __name__ == "__main__":
    normalize_coaches()