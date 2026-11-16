
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from importlib.util import spec_from_file_location, module_from_spec


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

# Reutilizar as funcoes do ficheiro 04
script_path = Path(__file__).parent / "04_two_stage_model.py"

spec = spec_from_file_location("two_stage", script_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

FEATURES = module.FEATURES


def evaluate(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def main():

    df = pd.read_csv(DATA_DIR / "pim_dataset.csv")

    train = df[df["year"] <= 2006].copy()
    test = df[df["year"].between(2007, 2009)].copy()

    print("\n=== AVALIACAO FINAL - PIM C ===")
    print("Treino:", len(train))
    print("Teste:", len(test))

    # Baseline
    baseline = (
        test["avg_PIM_3"]
        .fillna(test["prev_PIM"])
        .fillna(0)
        .clip(lower=0)
        .to_numpy()
    )

    # Random Forest direto
    direct = module.regression_model()
    direct.fit(
        train[FEATURES],
        train["target_PIM"],
    )

    pred_direct = np.maximum(
        0,
        direct.predict(test[FEATURES]),
    )

    # Classificador de participacao
    classifier = module.classification_model()

    classifier.fit(
        train[FEATURES],
        train["played_next_year"].astype(int),
    )

    probability = classifier.predict_proba(
        test[FEATURES]
    )[:, 1]

    # Regressor condicional
    played_train = train[
        train["played_next_year"] == True
    ]

    conditional = module.regression_model()

    conditional.fit(
        played_train[FEATURES],
        played_train["target_PIM"],
    )

    pred_conditional = np.maximum(
        0,
        conditional.predict(test[FEATURES]),
    )

    pred_two_stage = probability * pred_conditional

    # Resultados
    y = test["target_PIM"].to_numpy()

    results = []

    for name, pred in [
        ("Baseline", baseline),
        ("RandomForest direto", pred_direct),
        ("Modelo duas fases", pred_two_stage),
    ]:
        results.append({
            "Modelo": name,
            **evaluate(y, pred),
        })

    results_df = pd.DataFrame(results)

    print("\n=== RESULTADOS ===")
    print(results_df.round(3).to_string(index=False))

    print(
        "\nAUC participacao:",
        round(
            roc_auc_score(
                test["played_next_year"].astype(int),
                probability,
            ),
            3,
        ),
    )

    # Avaliacao por epoca
    predictions = test[
        [
            "playerID",
            "year",
            "pos",
            "target_PIM",
            "played_next_year",
        ]
    ].copy()

    predictions["pred_baseline"] = baseline
    predictions["pred_direct"] = pred_direct
    predictions["prob_participation"] = probability
    predictions["pred_conditional"] = pred_conditional
    predictions["pred_two_stage"] = pred_two_stage

    print("\n=== MAE POR EPOCA ===")

    for year, group in predictions.groupby("year"):
        print(
            f"{year}: "
            f"{mean_absolute_error(group['target_PIM'], group['pred_two_stage']):.3f}"
        )

    results_df.to_csv(
        DATA_DIR / "pim_final_comparison.csv",
        index=False,
    )

    predictions.to_csv(
        DATA_DIR / "pim_final_two_stage_predictions.csv",
        index=False,
    )

    print("\nResultados guardados com sucesso.")


if __name__ == "__main__":
    main()
