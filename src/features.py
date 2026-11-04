"""Leitura dos dados e features pré-época.

1. load_tables / team_season_stats: estatísticas de cada equipa-época que
   descrevem a PRÓPRIA época (taxas por jogo, valores relativos à liga, ...).
2. add_lag_features e seguintes: desfasam essas estatísticas — para a época t
   só se usam épocas < t, porque na véspera da época t só se conhece até t-1.
"""

import numpy as np
import pandas as pd

from .config import DATA_DIR, FIRST_TRAIN_SEASON, LEAGUE, N_LAGS

# ---------------------------------------------------------------------------
# Leitura e estatísticas por equipa-época
# ---------------------------------------------------------------------------
DEFAULT_TABLES = ["Teams", "TeamSplits", "Scoring", "Goalies", "Master"]

PLAYOFF_DEPTH = {"PRE": 0.5, "DSF": 1, "CQF": 1, "R1": 1, "QF": 1,
                 "DF": 2, "CSF": 2, "R2": 2, "SF": 2,
                 "CF": 3, "R3": 3, "F": 4, "SC": 5}


def load_tables(names=DEFAULT_TABLES, league=LEAGUE):
    """Lê as tabelas pedidas e filtra pela liga (quando têm coluna lgID)."""
    tables = {}
    for name in names:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro em falta: {path}")
        df = pd.read_csv(path, low_memory=False)
        if "lgID" in df:
            df = df.loc[df["lgID"].eq(league)].copy()
        tables[name] = df
    return tables


def _base_team_stats(teams):
    """Taxas por jogo e valores relativos à média da liga (normalização por época)."""
    df = teams[["year", "tmID", "franchID", "confID", "divID", "rank",
                "playoff", "G", "W", "L", "T", "OTL", "Pts", "GF", "GA",
                "PIM", "PPG", "PPC", "PKG", "PKC", "SHF", "SoW", "SoL"]].copy()
    g = df["G"].replace(0, np.nan)

    df["pts_pct"] = df["Pts"] / (2 * g)
    # Relativo à média da liga: remove a inflação de pontos pós-2005 (OTL).
    df["pts_pct_rel"] = df["pts_pct"] - df.groupby("year")["pts_pct"].transform("mean")
    df["win_pct"] = df["W"] / g
    df["gf_pg"] = df["GF"] / g
    df["ga_pg"] = df["GA"] / g
    df["gd_pg"] = (df["GF"] - df["GA"]) / g
    df["gf_rel"] = df["gf_pg"] - df.groupby("year")["gf_pg"].transform("mean")
    df["ga_rel"] = df["ga_pg"] - df.groupby("year")["ga_pg"].transform("mean")
    # "Pitagórica": percentagem esperada pelos golos (menos ruído do que vitórias).
    df["pyth"] = df["GF"] ** 2 / (df["GF"] ** 2 + df["GA"] ** 2)
    df["luck"] = df["win_pct"] - df["pyth"]
    df["pp_pct"] = df["PPG"] / df["PPC"].replace(0, np.nan)
    df["pk_pct"] = 1 - df["PKG"] / df["PKC"].replace(0, np.nan)
    df["pim_pg"] = df["PIM"] / g
    df["playoff_depth"] = df["playoff"].map(PLAYOFF_DEPTH).fillna(0)
    div_size = df.groupby(["year", "divID"])["tmID"].transform("count")
    df["rank_norm"] = (df["rank"] - 1) / (div_size - 1).replace(0, np.nan)
    return df


def _split_stats(splits):
    """Forma no fim da época (fev–abr) e desempenho casa/fora."""
    s = splits.copy()
    late = {"W": 0, "L": 0, "T": 0, "OL": 0}
    for m in ["Feb", "Mar", "Apr"]:
        for k in late:
            late[k] = late[k] + s.get(f"{m}{k}", pd.Series(0, index=s.index)).fillna(0)
    late_g = late["W"] + late["L"] + late["T"] + late["OL"]
    s["late_pts_pct"] = (2 * late["W"] + late["T"] + late["OL"]) / (2 * late_g.replace(0, np.nan))
    home_g = s[["hW", "hL", "hT", "hOTL"]].fillna(0).sum(axis=1)
    road_g = s[["rW", "rL", "rT", "rOTL"]].fillna(0).sum(axis=1)
    s["home_win_pct"] = s["hW"] / home_g.replace(0, np.nan)
    s["road_win_pct"] = s["rW"] / road_g.replace(0, np.nan)
    return s[["year", "tmID", "late_pts_pct", "home_win_pct", "road_win_pct"]]


def _roster_stats(scoring, master):
    """Idade média do plantel (ponderada por jogos) e concentração de pontos."""
    sc = scoring.merge(master[["playerID", "birthYear"]], on="playerID", how="left")
    sc["age"] = sc["year"] - sc["birthYear"]
    sc["GPw"] = sc["GP"].fillna(0)
    sc["age_w"] = sc["age"] * sc["GPw"]
    sc["old"] = (sc["age"] >= 32) * sc["GPw"]
    sc["young"] = (sc["age"] <= 23) * sc["GPw"]
    agg = sc.groupby(["year", "tmID"]).agg(age_w=("age_w", "sum"), gp=("GPw", "sum"),
                                           old=("old", "sum"), young=("young", "sum"))
    gp = agg["gp"].replace(0, np.nan)
    agg["roster_age"] = agg["age_w"] / gp
    agg["share_old"] = agg["old"] / gp
    agg["share_young"] = agg["young"] / gp

    def top3_share(pts):
        pts = pts.sort_values(ascending=False)
        return pts.head(3).sum() / pts.sum() if pts.sum() > 0 else np.nan

    agg["top3_share"] = sc.groupby(["year", "tmID"])["Pts"].apply(top3_share)
    return agg[["roster_age", "share_old", "share_young", "top3_share"]].reset_index()


def _goalie_stats(goalies):
    """Save% da equipa e do guarda-redes principal (mais minutos)."""
    go = goalies.copy()
    team = go.groupby(["year", "tmID"]).agg(GA_g=("GA", "sum"), SA_g=("SA", "sum"))
    team["team_sv"] = 1 - team["GA_g"] / team["SA_g"].replace(0, np.nan)
    team.loc[team["SA_g"] <= 0, "team_sv"] = np.nan

    main = go.sort_values("Min", ascending=False).drop_duplicates(["year", "tmID"])
    main = main.assign(main_sv=1 - main["GA"] / main["SA"].replace(0, np.nan))
    total_min = go.groupby(["year", "tmID"])["Min"].transform("sum")
    main["main_min_share"] = main["Min"] / total_min.loc[main.index]
    return (team[["team_sv"]].reset_index()
            .merge(main[["year", "tmID", "main_sv", "main_min_share"]],
                   on=["year", "tmID"], how="outer"))


def team_season_stats(tables):
    """Uma linha por equipa-época com todas as estatísticas da própria época."""
    df = _base_team_stats(tables["Teams"])
    df = df.merge(_split_stats(tables["TeamSplits"]), on=["year", "tmID"], how="left")
    df["late_form"] = df["late_pts_pct"] - df["pts_pct"]
    df = df.merge(_roster_stats(tables["Scoring"], tables["Master"]), on=["year", "tmID"], how="left")
    df = df.merge(_goalie_stats(tables["Goalies"]), on=["year", "tmID"], how="left")
    return df


# ---------------------------------------------------------------------------
# Features desfasadas (sem fuga de informação)
# ---------------------------------------------------------------------------
ID_COLS = ["year", "tmID", "franchID", "confID", "divID"]

# Estatísticas usadas da época anterior (lag 1)
STAT_COLS = ["pts_pct_rel", "win_pct", "gf_rel", "ga_rel", "gd_pg", "pyth", "luck",
             "pp_pct", "pk_pct", "pim_pg", "playoff_depth", "rank_norm",
             "late_pts_pct", "late_form", "home_win_pct", "road_win_pct",
             "roster_age", "share_old", "share_young", "top3_share",
             "team_sv", "main_sv", "main_min_share"]
# Estatísticas usadas das épocas t-2, t-3 (tendência de longo prazo)
LONG_LAG_COLS = ["pts_pct_rel", "gd_pg", "playoff_depth"]
LAG_WEIGHTS = [3, 2, 1]        # mais peso à época mais recente

TARGET_AND_LABELS = ["target", "rank", "Pts", "W"]


def add_lag_features(stats, target_rows):
    """Junta a cada linha (equipa na época t) as estatísticas das épocas anteriores.

    target_rows: colunas ID_COLS — estrutura da época t, conhecida antes do 1.º jogo.
    O lag é a época anterior *jogada* (2004 não existiu) da mesma franquia, ligada
    por franchID para sobreviver a mudanças de cidade (ex.: QUE -> COL).
    """
    hist = stats.set_index(["franchID", "year"]).sort_index()
    seasons = np.sort(stats["year"].unique())
    rows = []
    for _, r in target_rows.iterrows():
        prev_years = seasons[seasons < r["year"]][::-1][:N_LAGS]
        feat = {"year": r["year"], "tmID": r["tmID"]}
        for k, y in enumerate(prev_years, start=1):
            key = (r["franchID"], y)
            cols = STAT_COLS if k == 1 else LONG_LAG_COLS
            for c in cols:
                feat[f"{c}_l{k}"] = hist.at[key, c] if key in hist.index else np.nan
        feat.update(_history_summary(feat))
        rows.append(feat)
    out = target_rows.merge(pd.DataFrame(rows), on=["year", "tmID"])
    return _add_schedule_strength(out)


def _history_summary(feat):
    vals = np.array([feat.get(f"pts_pct_rel_l{k}", np.nan) for k in range(1, N_LAGS + 1)],
                    dtype=float)
    w = np.array(LAG_WEIGHTS[:N_LAGS], dtype=float)
    ok = ~np.isnan(vals)
    return {
        "pts_rel_wavg": (vals[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else np.nan,
        "trend": vals[0] - vals[1] if N_LAGS > 1 else np.nan,
        "seasons_hist": int(ok.sum()),
        "is_new": int(np.isnan(vals[0])),
    }


def _add_schedule_strength(out):
    """Força média (histórica) dos rivais de divisão e conferência."""
    for grp, name in [("divID", "div_rivals"), ("confID", "conf_rivals")]:
        total = out.groupby(["year", grp])["pts_rel_wavg"].transform("sum")
        count = out.groupby(["year", grp])["pts_rel_wavg"].transform("count")
        own = out["pts_rel_wavg"]
        out[f"{name}_strength"] = (total - own.fillna(0)) / (count - own.notna()).replace(0, np.nan)
    out["div_size"] = out.groupby(["year", "divID"])["tmID"].transform("count")
    return out


def build_training_set(stats):
    """Todas as equipa-épocas históricas com features desfasadas e o alvo."""
    target_rows = stats.loc[stats["year"] >= FIRST_TRAIN_SEASON, ID_COLS]
    data = add_lag_features(stats, target_rows)
    labels = stats[["year", "tmID", "pts_pct_rel", "rank", "Pts", "W"]]
    return data.merge(labels.rename(columns={"pts_pct_rel": "target"}), on=["year", "tmID"])


def build_target_season(stats, target_season):
    """Linhas da época-alvo, com o alinhamento conhecido antes da época.

    Se Teams.csv já tiver linhas da época-alvo (ex.: ficheiro de teste com equipas
    e divisões mas sem resultados), usam-se essas. Caso contrário assume-se o
    mesmo alinhamento da última época conhecida.
    """
    known = stats[stats["year"] == target_season]
    if len(known):
        rows = known[ID_COLS].copy()
    else:
        last = stats[stats["year"] == stats["year"].max()]
        rows = last[ID_COLS].assign(year=target_season)
    return add_lag_features(stats, rows)


def feature_columns(data):
    excluded = set(ID_COLS) | set(TARGET_AND_LABELS)
    return [c for c in data.columns if c not in excluded]
