## teams.csv

| Column | Meaning | Calculation |
|---|---|---|
| `year` | Season year |
| `lgID` | League identifier |
| `tmID` | Team identifier |
| `rank` | Historical team rank |
| `G` | Games played |
| `W` | Wins |
| `Pts` | Regular-season points |
| `GF` | Goals for |
| `GA` | Goals against |
| `points_per_game` | Points per game | `Pts / G` |
| `win_rate` | Win rate | `W / G` |
| `goals_for_per_game` | Goals scored per game | `GF / G` |
| `goals_against_per_game` | Goals conceded per game | `GA / G` |
| `goal_differential` | Goal differential | `GF - GA` |
| `goal_differential_per_game` | Goal differential per game | `(GF - GA) / G` |


## coaches.csv

| Column | Meaning | Calculation |
|---|---|---|
| `coachID` | Coach identifier |
| `year` | Season year |
| `tmID` | Team identifier |
| `lgID` | League identifier |
| `stint` | Coach's stint number with the team |
| `g` | Games coached |
| `w` | Wins while coaching |
| `postg` | Postseason games coached |
| `postw` | Postseason wins |
| `win_rate` | Coach win rate | `w / g` |
| `postseason_win_rate` | Coach postseason win rate | `postw / postg` |


## Possible baseline comparison metrics

### Team ranking

previous-season baseline
mean absolute error (MAE) (average distance between each predicted rank and actual rank)
spearman correlation (compares the order of the predicted ranking with the order of the actual ranking)

### Coaching changes

no-change baseline

precision (percentage of predicted coaching changes that happened)
recall (percentage of actual coaching changes that the model detected)
f1-score, (precision and recall combination)
