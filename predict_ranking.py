"""Tarefa (a): previsão da classificação final da época regular (NHL).

1. src/features.py    lê os CSV, calcula estatísticas por equipa-época e
2.                    desfasa-as: a época t só vê épocas < t (sem fuga);
3. src/models.py      prevê a percentagem de pontos relativa à liga;
4. src/evaluation.py  ordena dentro da divisão (definição de `rank` em Teams.csv)
                      e avalia com validação walk-forward.

Uso:  python predict_ranking.py
Saídas (pasta outputs/):
    ranking_validation_by_season.csv  métricas por época e modelo
    ranking_validation_summary.csv    médias por modelo
    predictions_<ano>_ranking.csv      previsão final para a época-alvo
"""

import warnings

from src.config import BASE_DIR, FIRST_EVAL_SEASON, GAMES_PER_SEASON, OUT_DIR
from src.evaluation import best_model, rank_within, summarize, walk_forward
from src.features import (build_target_season, build_training_set, feature_columns,
                          load_tables, team_season_stats)
from src.models import fit_predict

warnings.filterwarnings("ignore", category=RuntimeWarning)


def validate(data, features):
    metrics, _ = walk_forward(data, features)
    metrics.to_csv(OUT_DIR / "ranking_validation_by_season.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT_DIR / "ranking_validation_summary.csv")
    print(f"\nValidação walk-forward ({FIRST_EVAL_SEASON}-{data.year.max()}), médias por época:")
    print(summary.round(3).to_string())
    return best_model(summary)


def predict_season(stats, data, features, model_name):
    target_season = int(stats.loc[stats["rank"].notna(), "year"].max()) + 1
    target = build_target_season(stats, target_season)
    train = data[data["target"].notna()]
    target["pred_pts_pct_rel"] = fit_predict(model_name, train[features], train["target"],
                                             target[features])

    # Pontos esperados: média de pontos da última época + desvio previsto
    last_mean = stats.loc[stats["year"] == stats["year"].max(), "pts_pct"].mean()
    target["pred_points"] = ((target["pred_pts_pct_rel"] + last_mean) * 2 * GAMES_PER_SEASON).round(1)
    target["pred_rank"] = rank_within(target, "pred_pts_pct_rel", ("year", "divID"))
    target["pred_conf_rank"] = rank_within(target, "pred_pts_pct_rel", ("year", "confID"))
    target["pred_league_rank"] = rank_within(target, "pred_pts_pct_rel", ("year",))

    out = (target[["year", "tmID", "franchID", "confID", "divID", "pred_rank",
                   "pred_conf_rank", "pred_league_rank", "pred_points", "pred_pts_pct_rel"]]
           .sort_values(["confID", "divID", "pred_rank"]))
    out_file = OUT_DIR / f"predictions_{target_season}_ranking.csv"
    out.to_csv(out_file, index=False)
    print(f"\nPrevisão para {target_season} ({model_name}) -> {out_file.relative_to(BASE_DIR)}")
    print(out.drop(columns=["franchID", "pred_pts_pct_rel"]).to_string(index=False))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    stats = team_season_stats(load_tables())
    data = build_training_set(stats)
    features = feature_columns(data)
    print(f"Dataset: {len(data)} equipa-épocas ({data.year.min()}-{data.year.max()}), "
          f"{len(features)} features")

    model_name = validate(data, features)
    print(f"\nModelo escolhido (menor MAE de posição): {model_name}")
    predict_season(stats, data, features, model_name)


if __name__ == "__main__":
    main()
