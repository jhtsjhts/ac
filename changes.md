# changes

## 1. Number of teams

| Years | Teams |
|---|---|
| 1917 | 4 |
| 1918 | 3 |
| 1919-1923 | 4 |
| 1924 | 6 |
| 1925 | 7 |
| 1926-1930 | 10 |
| 1931 | 8 |
| 1932-1934 | 9 |
| 1935-1937 | 8 |
| 1938-1941 | 7 |
| 1942-1966 | 6 (the "Original Six" era) |
| 1967-1969 | 12 |
| 1970-1971 | 14 |
| 1972-1973 | 16 |
| 1974-1977 | 18 |
| 1978 | 17 |
| 1979-1990 | 21 |
| 1991 | 22 |
| 1992 | 24 |
| 1993-1997 | 26 |
| 1998 | 27 |
| 1999 | 28 |
| 2000-2009 | 30 |

## 2. Games per season

| Years | Games |
|---|---|
| 1917 | 22 (one team played only 6) |
| 1918 | 18 |
| 1919-1923 | 24 |
| 1924 | 30 |
| 1925 | 36 |
| 1926-1930 | 44 |
| 1931-1941 | 48 |
| 1942-1945 | 50 |
| 1946-1948 | 60 |
| 1949-1966 | 70 |
| 1967 | 74 |
| 1968-1969 | 76 |
| 1970-1973 | 78 |
| 1974-1991 | 80 |
| 1992-1993 | 84 |
| 1994 | 48 (lockout) |
| 1995-2009 | 82 |

Consequence: `Pts`, `W`, `GF` and `GA` must be normalised per game before comparing seasons.

## 3. Teams that made the playoffs

| Years | Playoff teams |
|---|---|
| 1917-1918 | 2 |
| 1919 | 1 (finals halted by the influenza epidemic, no Cup winner) |
| 1920-1923 | 2 |
| 1924-1925 | 3 |
| 1926-1941 | 6 |
| 1942-1966 | 4 |
| 1967-1973 | 8 |
| 1974-1978 | 12 |
| 1979-2009 | 16 |

## 4. How games end and how points work

| Years | What the data shows |
|---|---|
| 1917-1920 and 1923 | No tied games recorded |
| 1921-1922, 1924-1998 | Ties exist, no overtime losses |
| 1999-2003 | Ties plus overtime losses (`OTL` appears, the loser gets 1 point) |
| 2004 | No season |
| 2005-2009 | No ties; shootouts (`SoW`, `SoL`) appear; about 270-300 overtime losses per season |

- *(Background)* A regular-season overtime period was introduced around 1983, but a tie was still possible.
- Consequence: `T`, `OTL`, `SoW` and `SoL` are structurally missing or zero depending on the era. They are not data errors.

## 5. Goals per team per game, every season

| Years | Goals/game | Note |
|---|---|---|
| 1917-1921 | 4.40, 4.13, 4.79, 4.23, 3.96 | Tiny league, high scoring |
| 1922-1925 | 3.26, 2.66, 2.50, 2.31 | Falling |
| 1926-1928 | 2.00, 1.90, 1.46 | Lowest scoring in the data |
| 1929-1940 | 2.96, 2.40, 2.49, 2.28, 2.41, 2.52, 2.16, 2.46, 2.53, 2.53, 2.49, 2.68 | Stable and low |
| 1941-1946 | 3.12, 3.61, 4.08, 3.68, 3.34, 3.16 | Wartime peak in 1943 |
| 1947-1955 | 2.93, 2.72, 2.73, 2.71, 2.60, 2.40, 2.40, 2.52, 2.53 | Defensive |
| 1956-1966 | 2.69, 2.80, 2.90, 2.95, 3.00, 3.01, 2.97, 2.78, 2.88, 3.04, 2.98 | Slowly rising |
| 1967-1973 | 2.79, 2.98, 2.90, 3.12, 3.07, 3.28, 3.20 | Expansion era |
| 1974-1979 | 3.42, 3.41, 3.32, 3.30, 3.50, 3.51 | High |
| 1980-1985 | 3.84, 4.01, 3.86, 3.94, 3.89, 3.97 | Peak (4.01 in 1981) |
| 1986-1992 | 3.67, 3.71, 3.74, 3.68, 3.46, 3.48, 3.63 | Declining |
| 1993-1998 | 3.24, 2.99, 3.14, 2.92, 2.64, 2.63 | Sharp drop |
| 1999-2003 | 2.75, 2.76, 2.62, 2.65, 2.57 | "Dead puck era" (low point 2.57 in 2003) |
| 2005-2009 | 3.08, 2.95, 2.78, 2.91, 2.84 | After the lockout |

- *(Background)* Defensive trapping systems and larger goalie equipment explain the 1996-2003 drop. After the lockout, rule changes (obstruction calls, removing the two-line pass, goalie equipment limits) aimed to raise scoring.

## 6. Conferences and divisions in the data

| Years | Structure |
|---|---|
| 1917-1925 | No conferences or divisions |
| 1926-1937 | 2 divisions |
| 1938-1966 | No conferences or divisions |
| 1967-1973 | 2 divisions, no conferences |
| 1974-1992 | 2 conferences (Campbell and Wales), 4 divisions (until 1997) |
| 1993-1997 | 2 conferences (Eastern and Western), 4 divisions |
| 1998-2009 | 2 conferences (Eastern and Western), 6 divisions |

## 7. Meaning of the `rank` column

| Years | `rank` is |
|---|---|
| 1917 | Duplicated within the season (split season) |
| 1918-1925 | Whole-league rank |
| 1926-1937 | Rank within a division |
| 1938-1966 | Whole-league rank |
| 1967-2003, 2005-2009 | Rank within a division |

Consequence: from 1967 onward, `rank` cannot be used directly as target (a). A league-wide order must be built from points.

## 8. Economics and players *(background)*

- Free agency arrived gradually, and the 2005 salary cap forced parity. After the cap, the gap between the best and worst teams likely narrowed, which matters for target (a).
- Early players were almost entirely Canadian. American and then European players arrived from the 1980s and 1990s.
- Helmets became compulsory for new players in 1979.
- Recorded statistics change over time: `+/-` is missing for 54% of `Scoring` rows in the 1960s and about 12% in the 1970s, and is almost always present from the 1980s.

## What this means for the model

- Earlier eras differ from the 30-team, 82-game, post-2005 league being predicted. The training window is a trade-off between more data and more relevant data.
- Window options:

| Window | Seasons | Trade-off |
|---|---|---|
| 2005-2009 | 5 | Shootout and cap era, but too few seasons |
| 1995-2009 (without 2004) | 14 | 82 games and a stable format, but includes the dead-puck period |
| 1979-2009 (without 2004) | 30 | 16-team playoffs and 21+ teams, but includes ties and several scoring regimes |

- Whatever the window, normalise per game and per season average, and consider an era indicator.