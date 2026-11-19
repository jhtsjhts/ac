"""Compare coaching-change models and training windows with temporal validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from coach_model import (
    FEATURES, IDENTIFIERS, ROOT, build_dataset, classification_metrics,
    load_data, prepare_history,
)


OBJECTIVES = {"f1": 1.0, "f0.5": 0.5}
THRESHOLDS = np.linspace(0, 1, 41)


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    window: int | None
    class_weight: str

    @property
    def training_window(self) -> str:
        return "all_history" if self.window is None else f"last_{self.window}_seasons"

    @property
    def candidate_id(self) -> str:
        return f"{self.algorithm}__{self.training_window}__{self.class_weight}"

    def metadata(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "algorithm": self.algorithm,
            "training_window": self.training_window, "class_weight": self.class_weight,
        }


def candidates() -> list[Candidate]:
    """A fixed factorial experiment: 3 algorithms x 2 windows x 2 weights."""
    return [
        Candidate(algorithm, window, weight)
        for algorithm in ["logistic_regression", "random_forest", "gradient_boosting"]
        for window in [None, 20]
        for weight in ["none", "balanced"]
    ]


def make_estimator(candidate: Candidate) -> Pipeline:
    weight = None if candidate.class_weight == "none" else candidate.class_weight
    if candidate.algorithm == "logistic_regression":
        # Keep C from the original model's pre-2009 validation; do not tune it
        # again while changing the algorithm, window and weighting scheme.
        classifier = LogisticRegression(C=0.01, class_weight=weight, max_iter=3000, random_state=42)
    elif candidate.algorithm == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=8,
            max_features="sqrt", class_weight=weight, random_state=42, n_jobs=1,
        )
    elif candidate.algorithm == "gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=15, l2_regularization=1.0, early_stopping=False,
            class_weight=weight, random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {candidate.algorithm}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler() if candidate.algorithm == "logistic_regression" else "passthrough"),
        ("classifier", classifier),
    ])


def training_rows(dataset: pd.DataFrame, cutoff: int, window: int | None) -> pd.DataFrame:
    """Keep whole seasons strictly before cutoff; count seasons, not calendar years."""
    past = dataset.loc[dataset["year"].lt(cutoff)]
    years = sorted(past["year"].unique())
    if window is not None:
        if window < 1:
            raise ValueError("Training window must be positive.")
        years = years[-window:]
    return past.loc[past["year"].isin(years) & past["actual_coach_change"].notna()].copy()


def fit(model: Pipeline, train: pd.DataFrame) -> None:
    if train["actual_coach_change"].nunique() != 2:
        raise ValueError("Each training split needs examples of both classes.")
    model.fit(train[FEATURES], train["actual_coach_change"].astype(int))


def choose_threshold(actual, probabilities, beta: float) -> tuple[float, float]:
    if len(np.unique(actual)) != 2:
        raise ValueError("Threshold selection needs both classes; increase --calibration-seasons.")
    options = []
    for threshold in THRESHOLDS:
        predicted = probabilities >= threshold
        options.append((
            float(fbeta_score(actual, predicted, beta=beta, zero_division=0)),
            float(precision_score(actual, predicted, zero_division=0)),
            -abs(float(threshold) - 0.5), float(threshold),
        ))
    selected = max(options)
    return selected[3], selected[0]


def fit_for_year(dataset, candidate, year, calibration_seasons):
    """Tune thresholds on earlier years, then refit before predicting this year.

    For validation year 2003 with 3 calibration seasons: fit before 2000,
    tune on 2000/2001/2002, then refit on all eligible seasons before 2003.
    Each fit independently applies the candidate's training window.
    """
    if calibration_seasons < 1:
        raise ValueError("Calibration seasons must be positive.")
    past = training_rows(dataset, year, None)
    years = sorted(past["year"].unique())
    if len(years) < calibration_seasons + 5:
        raise ValueError("Need at least five training seasons plus the threshold-calibration seasons before every validation year.")
    calibration_years = years[-calibration_seasons:]
    earlier = training_rows(dataset, int(calibration_years[0]), candidate.window)
    calibration = past.loc[past["year"].isin(calibration_years)]
    calibration_model = make_estimator(candidate)
    fit(calibration_model, earlier)
    probabilities = calibration_model.predict_proba(calibration[FEATURES])[:, 1]
    thresholds, calibration_scores = {}, {}
    for objective, beta in OBJECTIVES.items():
        thresholds[objective], calibration_scores[objective] = choose_threshold(
            calibration["actual_coach_change"].astype(int), probabilities, beta,
        )
    train = training_rows(dataset, year, candidate.window)
    model = make_estimator(candidate)
    fit(model, train)
    split = {
        "threshold_training_years": [int(value) for value in sorted(earlier["year"].unique())],
        "threshold_calibration_years": [int(value) for value in calibration_years],
        "final_training_years": [int(value) for value in sorted(train["year"].unique())],
        "threshold_training_rows": len(earlier), "threshold_calibration_rows": len(calibration),
        "final_training_rows": len(train), "prediction_year": int(year),
    }
    return model, thresholds, calibration_scores, split


def metrics(actual, predicted, probabilities=None) -> dict:
    result = classification_metrics(actual, predicted, probabilities)
    tn, fp, fn, tp = np.asarray(result.pop("confusion_matrix")).ravel()
    result.update({"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)})
    result["f0.5"] = float(fbeta_score(actual, predicted, beta=0.5, zero_division=0))
    return result


def run_experiment(dataset, reserved_year, validation_seasons=5, calibration_seasons=3, configs=None, verbose=False):
    """Select models using earlier seasons only; never evaluate the reserved year."""
    configs = candidates() if configs is None else configs
    past = dataset.loc[dataset["year"].lt(reserved_year)].copy()
    if past.duplicated(["year", "lgID", "tmID"]).any():
        raise ValueError("Experiment dataset contains duplicate team-season rows.")
    years = sorted(past.loc[past["actual_coach_change"].notna(), "year"].unique())
    if validation_seasons < 1 or len(years) < validation_seasons + calibration_seasons + 5:
        raise ValueError("Not enough earlier seasons for the requested validation and calibration windows.")
    if not configs:
        raise ValueError("No candidate configurations supplied.")
    validation_years = years[-validation_seasons:]
    predictions, fold_rows, splits = [], [], []
    with threadpool_limits(limits=1):
        for index, candidate in enumerate(configs, start=1):
            for year in validation_years:
                model, thresholds, calibration_scores, split = fit_for_year(past, candidate, int(year), calibration_seasons)
                validation = past.loc[past["year"].eq(year) & past["actual_coach_change"].notna()]
                probabilities = model.predict_proba(validation[FEATURES])[:, 1]
                splits.append({**candidate.metadata(), **split})
                for objective, threshold in thresholds.items():
                    rows = validation[IDENTIFIERS + ["actual_coach_change"]].copy()
                    rows["actual_coach_change"] = rows["actual_coach_change"].astype(int)
                    for key, value in candidate.metadata().items():
                        rows[key] = value
                    rows["objective"] = objective
                    rows["threshold"] = threshold
                    rows["change_probability"] = probabilities
                    rows["predicted_coach_change"] = (probabilities >= threshold).astype(int)
                    predictions.append(rows)
                    fold_rows.append({
                        **candidate.metadata(), "objective": objective, "year": int(year),
                        "threshold": threshold, "threshold_calibration_score": calibration_scores[objective],
                        "training_first_year": split["final_training_years"][0],
                        "training_last_year": split["final_training_years"][-1],
                        "training_seasons": len(split["final_training_years"]),
                        "training_rows": split["final_training_rows"],
                        **metrics(rows["actual_coach_change"].to_numpy(), rows["predicted_coach_change"], probabilities),
                    })
            if verbose:
                print(f"[{index}/{len(configs)}] {candidate.candidate_id}: {len(validation_years)} validation seasons completed", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    summary = []
    for (candidate_id, objective), group in predictions.groupby(["candidate_id", "objective"], sort=False):
        per_year = folds.loc[folds["candidate_id"].eq(candidate_id) & folds["objective"].eq(objective)]
        summary.append({
            **{key: group.iloc[0][key] for key in ["candidate_id", "algorithm", "training_window", "class_weight"]},
            "objective": objective,
            **metrics(group["actual_coach_change"].to_numpy(), group["predicted_coach_change"], group["change_probability"]),
            "mean_season_f1": float(per_year["f1"].mean()),
            "std_season_f1": float(per_year["f1"].std(ddof=0)),
            "mean_season_average_precision": float(per_year["average_precision"].mean()),
        })
    comparison = pd.DataFrame(summary)
    selected = {}
    for objective in OBJECTIVES:
        ranked = comparison.loc[comparison["objective"].eq(objective)].sort_values(
            [objective, "mean_season_average_precision", "precision", "candidate_id"],
            ascending=[False, False, False, True], kind="stable",
        )
        winner = ranked.iloc[0]
        config = next(item for item in configs if item.candidate_id == winner["candidate_id"])
        selected[objective] = {
            "configuration": asdict(config), "candidate_id": config.candidate_id,
            "validation_precision": float(winner["precision"]), "validation_recall": float(winner["recall"]),
            "validation_f1": float(winner["f1"]), "validation_f0.5": float(winner["f0.5"]),
            "validation_average_precision": float(winner["average_precision"]),
        }
    baseline_data = past.loc[past["year"].isin(validation_years) & past["actual_coach_change"].notna()]
    actual = baseline_data["actual_coach_change"].to_numpy(dtype=int)
    baselines = pd.DataFrame([
        {"baseline": name, **metrics(actual, predicted)}
        for name, predicted in [
            ("no_change", np.zeros(len(actual), dtype=int)),
            ("all_change", np.ones(len(actual), dtype=int)),
            ("below_median", baseline_data["previous_points_vs_median"].lt(0).to_numpy(dtype=int)),
        ]
    ])
    metadata = {
        "reserved_year": int(reserved_year), "reserved_year_used_for_selection": False,
        "validation_years": [int(value) for value in validation_years],
        "threshold_calibration_seasons": calibration_seasons,
        "threshold_grid": THRESHOLDS.tolist(), "features": FEATURES,
        "selection_rule": "Pooled validation F-beta, then mean per-season average precision, then precision, then candidate ID.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "classifier_parameters": {
            item.candidate_id: make_estimator(item).named_steps["classifier"].get_params()
            for item in configs
        },
        "splits": splits,
    }
    return comparison, folds, predictions, baselines, selected, metadata


def markdown_table(frame, columns):
    def value(item):
        if isinstance(item, (float, np.floating)):
            return f"{item:.3f}" if pd.notna(item) else "n/a"
        return str(item)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(value(row[column]) for column in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def write_report(output_dir, comparison, baselines, selected, metadata, test_results=None):
    lines = [
        "# Experiência: modelos e períodos de treino", "",
        f"Épocas de validação: {metadata['validation_years']}. A época {metadata['reserved_year']} não entra na seleção.", "",
        "Mantiveram-se as mesmas 19 variáveis e a mesma definição de mudança de treinador. Foram comparadas 12 configurações: três algoritmos, dois períodos de treino e dois pesos de classe. Os parâmetros das árvores e C=0.01 da regressão logística foram fixados antes da comparação.", "",
        "Em cada época de validação, o limiar é escolhido num bloco anterior de três épocas (ou no número indicado na execução). O modelo usado para escolher esse limiar treina em épocas ainda anteriores. Depois há novo treino até à véspera da época de validação, respeitando a janela de cada configuração. F1 e F0,5 escolhem limiares separadamente, sem treinos adicionais. Não se trata de calibração de probabilidades.", "",
        "A janela de 20 conta épocas disponíveis, incluindo interrupções no calendário. Limita os exemplos usados em cada ajuste; as variáveis de carreira continuam a usar o histórico que já era conhecido. A imputação e a padronização são ajustadas de novo apenas no treino de cada divisão.", "",
        "As métricas seguintes agregam as previsões das épocas de validação. São resultados usados para escolher configurações; não são uma avaliação independente da configuração vencedora. A variação do F1 por época está em comparison.csv e folds.csv.", "",
    ]
    columns = ["algorithm", "training_window", "class_weight", "precision", "recall", "f1", "f0.5", "false_positives", "average_precision"]
    for objective in OBJECTIVES:
        ranked = comparison.loc[comparison["objective"].eq(objective)].sort_values([objective, "mean_season_average_precision", "precision", "candidate_id"], ascending=[False, False, False, True])
        lines.extend([
            f"## Limiares escolhidos por {objective}", "",
            markdown_table(ranked, columns), "",
            f"Configuração selecionada: **{selected[objective]['candidate_id']}**.", "",
        ])
    lines.extend([
        "## Referências nas mesmas épocas", "",
        markdown_table(baselines, ["baseline", "precision", "recall", "f1", "f0.5", "false_positives"]), "",
        "Os números de falsos positivos são totais das épocas de validação, não de uma só época. class_weight=balanced altera os pesos do treino e as probabilidades obtidas não foram calibradas. Modelos com limiares diferentes podem ter F1 diferente e a mesma capacidade de ordenar riscos; average precision ajuda a separar estas duas questões.", "",
        "Os ficheiros selection.json e protocol.json registam a escolha e todas as divisões temporais. Se a execução incluir --evaluate-test, test_metrics.json e test_predictions_*.csv contêm a avaliação posterior das configurações já escolhidas. Esta avaliação nunca altera o vencedor. Como 2009 já tinha sido analisada no desenvolvimento, os resultados nessa época devem ser tratados como uma comparação exploratória.", "",
    ])
    if test_results is not None:
        test_rows = [
            {"objective": objective, "candidate_id": result["candidate_id"], "threshold": result["threshold"], **result["metrics"]}
            for objective, result in test_results.items() if result["metrics"] is not None
        ]
        lines.extend([f"## Avaliação posterior em {metadata['reserved_year']}", ""])
        if test_rows:
            lines.extend([
                markdown_table(pd.DataFrame(test_rows), ["objective", "precision", "recall", "f1", "false_positives", "false_negatives", "average_precision", "roc_auc"]), "",
                "As configurações já estavam escolhidas e os limiares foram aprendidos apenas em épocas anteriores. A avaliação acima não altera a seleção. A qualidade da ordenação do risco (average precision e ROC AUC) e a qualidade das decisões no limiar escolhido (precision, recall e F1) devem ser analisadas separadamente. Uma época com poucas mudanças reais dá uma estimativa incerta do desempenho futuro.", "",
            ])
        else:
            lines.extend(["A época reservada não tem alvos conhecidos para calcular métricas.", ""])
    else:
        lines.extend(["A época reservada não foi avaliada nesta execução. Ficheiros de teste de execuções anteriores, se existirem no diretório, não pertencem a esta comparação.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_selected(dataset, reserved_year, selected, calibration_seasons, output_dir):
    """Evaluate only the frozen validation winners; do not rank by test results."""
    target = dataset.loc[dataset["year"].eq(reserved_year)].copy()
    if target.empty:
        raise ValueError("Reserved season has no teams to evaluate.")
    results = {}
    with threadpool_limits(limits=1):
        for objective, choice in selected.items():
            candidate = Candidate(**choice["configuration"])
            model, thresholds, calibration_scores, split = fit_for_year(dataset, candidate, reserved_year, calibration_seasons)
            predictions = target[IDENTIFIERS + ["actual_coach_change"]].copy()
            predictions["change_probability"] = model.predict_proba(target[FEATURES])[:, 1]
            predictions["threshold"] = thresholds[objective]
            predictions["predicted_coach_change"] = predictions["change_probability"].ge(thresholds[objective]).astype(int)
            predictions = predictions.sort_values("change_probability", ascending=False)
            predictions.to_csv(output_dir / f"test_predictions_{objective}.csv", index=False)
            known = predictions.loc[predictions["actual_coach_change"].notna()]
            results[objective] = {
                "candidate_id": candidate.candidate_id, "threshold": thresholds[objective],
                "threshold_calibration_score": calibration_scores[objective], "split": split,
                "teams_without_labels": int(predictions["actual_coach_change"].isna().sum()),
                "metrics": metrics(known["actual_coach_change"].to_numpy(dtype=int), known["predicted_coach_change"], known["change_probability"]) if not known.empty else None,
            }
            joblib.dump({
                "pipeline": model, "features": FEATURES, "threshold": thresholds[objective],
                "target_year": reserved_year, "training_last_year": split["final_training_years"][-1],
                "candidate": asdict(candidate), "objective": objective,
            }, output_dir / f"selected_model_{objective}.joblib")
    return results


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2009, help="Reserved season, excluded from model selection.")
    parser.add_argument("--league", default="NHL")
    parser.add_argument("--validation-seasons", type=int, default=5)
    parser.add_argument("--calibration-seasons", type=int, default=3)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate validation winners on the reserved season after saving selection.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        teams, coaches = load_data(ROOT / "raw_data")
        # Exclude reserved-season and later outcomes before building any features.
        seasons, earlier_coaches = prepare_history(teams.loc[teams["year"].lt(args.year)], coaches.loc[coaches["year"].lt(args.year)], args.league)
        dataset = build_dataset(seasons, earlier_coaches, seasons[IDENTIFIERS])
        comparison, folds, predictions, baselines, selected, metadata = run_experiment(
            dataset, args.year, args.validation_seasons, args.calibration_seasons, verbose=True,
        )
        metadata["test_evaluation_requested"] = args.evaluate_test
        output_dir = args.output_dir or ROOT / "outputs" / "coach_experiment" / args.league / str(args.year)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_dir / "comparison.csv", index=False)
        folds.to_csv(output_dir / "folds.csv", index=False)
        predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
        baselines.to_csv(output_dir / "baselines.csv", index=False)
        save_json(output_dir / "selection.json", selected)
        save_json(output_dir / "protocol.json", metadata)
        write_report(output_dir, comparison, baselines, selected, metadata)
        print("\nValidation comparison (thresholds selected by F1)")
        columns = ["algorithm", "training_window", "class_weight", "precision", "recall", "f1", "false_positives"]
        print(comparison.loc[comparison["objective"].eq("f1"), columns].sort_values("f1", ascending=False).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        for objective, choice in selected.items():
            print(f"Selected by {objective}: {choice['candidate_id']}")
        if args.evaluate_test:
            # Selection has already been frozen and saved. Only now access the
            # reserved season's outcomes, for the two selected configurations.
            seasons, current_coaches = prepare_history(teams.loc[teams["year"].le(args.year)], coaches.loc[coaches["year"].le(args.year)], args.league)
            complete = build_dataset(seasons, current_coaches, seasons[IDENTIFIERS])
            results = evaluate_selected(complete, args.year, selected, args.calibration_seasons, output_dir)
            save_json(output_dir / "test_metrics.json", results)
            write_report(output_dir, comparison, baselines, selected, metadata, results)
            print(f"\nReserved season {args.year} (exploratory evaluation after selection)")
            for objective, result in results.items():
                if result["metrics"] is not None:
                    score = result["metrics"]
                    print(f"{objective}: precision={score['precision']:.3f}, recall={score['recall']:.3f}, F1={score['f1']:.3f}, false positives={score['false_positives']}")
        print(f"\nResults saved to: {output_dir}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
