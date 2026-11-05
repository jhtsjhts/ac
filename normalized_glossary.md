## teams.csv

| Column | Meaning | Calculation |
|---|---|---|
| `year` | Season year | Source value |
| `lgID` | League identifier | Source value |
| `tmID` | Team identifier | Source value |
| `rank` | Historical team rank | Source value; not a universal target |
| `G` | Games played | Source value |
| `W` | Wins | Source value |
| `Pts` | Regular-season points | Source value |
| `GF` | Goals for | Source value |
| `GA` | Goals against | Source value |
| `points_per_game` | Points per game | `Pts / G` |
| `win_rate` | Win rate | `W / G` |
| `goals_for_per_game` | Goals scored per game | `GF / G` |
| `goals_against_per_game` | Goals conceded per game | `GA / G` |
| `goal_differential` | Goal differential | `GF - GA` |
| `goal_differential_per_game` | Goal differential per game | `(GF - GA) / G` |


## coaches.csv

| Column | Meaning | Calculation |
|---|---|---|
| `coachID` | Coach identifier | Source value |
| `year` | Season year | Source value |
| `tmID` | Team identifier | Source value |
| `lgID` | League identifier | Source value |
| `stint` | Coach's stint number with the team | Source value |
| `g` | Games coached | Source value |
| `w` | Wins while coaching | Source value |
| `postg` | Postseason games coached | Source value |
| `postw` | Postseason wins | Source value |
| `win_rate` | Coach win rate | `w / g` |
| `postseason_win_rate` | Coach postseason win rate | `postw / postg` |
