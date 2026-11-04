"""Conversão de scores em classificação, métricas e validação walk-forward."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FIRST_EVAL_SEASON
from .models import fit_predict_all

BASELINES = {
    # Classificação da época anterior, reaplicada à divisão atual
    "baseline_prev": "pts_pct_rel_l1",
    # Média ponderada das últimas 3 épocas
    "baseline_wavg": "pts_rel_wavg",
}
METRIC_COLS = ["mae_rank", "exact_acc", "within1_acc", "spearman_div",
               "div_winner_acc", "rmse_pts_pct"]


def rank_within(df, score_col, group_cols=("year", "divID")):
    """Posições distintas 1..n dentro de cada grupo (maior score = 1.º)."""
    return (df.groupby(list(group_cols))[score_col]
              .rank(ascending=False, method="first").astype(int))


def ranking_metrics(df, pred_rank_col):
    err = (df[pred_rank_col] - df["rank"]).abs()
    rhos = []
    for _, g in df.groupby("divID"):
        if len(g) > 2:
            rho = spearmanr(g[pred_rank_col], g["rank"]).correlation
            if not np.isnan(rho):
                rhos.append(rho)
    return {"mae_rank": err.mean(), "exact_acc": (err == 0).mean(),
            "within1_acc": (err <= 1).mean(),
            "spearman_div": np.mean(rhos) if rhos else np.nan,
            "div_winner_acc": df.loc[df[pred_rank_col] == 1, "rank"].eq(1).mean()}


def walk_forward(data, features, first_season=FIRST_EVAL_SEASON):
    """Para cada época t: treina com épocas < t e avalia em t."""
    seasons = sorted(y for y in data["year"].unique() if y >= first_season)
    records, all_preds = [], []
    for season in seasons:
        train = data[(data["year"] < season) & data["target"].notna()]
        test = data[data["year"] == season].copy()

        scores = {name: test[col].fillna(-1).to_numpy() for name, col in BASELINES.items()}
        scores.update(fit_predict_all(train[features], train["target"], test[features]))

        for name, score in scores.items():
            test[f"score_{name}"] = score
            test[f"rank_{name}"] = rank_within(test, f"score_{name}")
            m = ranking_metrics(test, f"rank_{name}")
            if name not in BASELINES:
                m["rmse_pts_pct"] = np.sqrt(((score - test["target"]) ** 2).mean())
            m.update(season=season, model=name)
            records.append(m)
        all_preds.append(test)
    return pd.DataFrame(records), pd.concat(all_preds)


def summarize(metrics):
    return (metrics.groupby("model")[METRIC_COLS].mean().sort_values("mae_rank"))


def best_model(summary):
    return summary.drop(index=list(BASELINES)).index[0]
