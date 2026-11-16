
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# CAMINHOS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

INPUT_FILE = DATA_DIR / "pim_dataset.csv"

# FEATURES
NUMERIC_FEATURES = [
    "prev_PIM",
    "prev_GP",
    "prev_pim_per_game",
    "avg_PIM_3",
    "avg_GP_3",
    "avg_pim_per_game_3",
    "experience",
]

CATEGORICAL_FEATURES = ["pos"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# AVALIACAO
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
    }


def main():

    print("\n=== PIM C - PRIMEIRA AVALIACAO ===")

    df = pd.read_csv(INPUT_FILE)

    train = df[df["year"] <= 2006].copy()
    test = df[df["year"].between(2007, 2009)].copy()

    print("Treino:", len(train))
    print("Teste:", len(test))

    X_train = train[FEATURES]
    y_train = train["target_PIM"]

    X_test = test[FEATURES]
    y_test = test["target_PIM"]

    # BASELINE
    baseline = (
        test["avg_PIM_3"]
        .fillna(test["prev_PIM"])
        .fillna(0)
        .clip(lower=0)
    )

    baseline_metrics = evaluate(
        y_test, baseline
    )

    # PRE-PROCESSAMENTO
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Unknown",
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # RANDOM FOREST
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=10,
                    max_features=0.7,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nA treinar Random Forest...")

    model.fit(X_train, y_train)

    predictions = np.maximum(
        0,
        model.predict(X_test),
    )

    model_metrics = evaluate(
        y_test, predictions
    )

    # RESULTADOS
    results = pd.DataFrame([
        {
            "Modelo": "Baseline_3_Years",
            **baseline_metrics,
        },
        {
            "Modelo": "RandomForest",
            **model_metrics,
        },
    ])

    print("\n=== RESULTADOS ===")
    print(results.round(3).to_string(index=False))

    improvement = (
        1
        - model_metrics["MAE"]
        / baseline_metrics["MAE"]
    ) * 100

    print(
        f"\nMelhoria MAE: {improvement:.2f}%"
    )

    # GUARDAR
    predictions_df = test[
        [
            "playerID",
            "year",
            "pos",
            "played_next_year",
            "target_PIM",
        ]
    ].copy()

    predictions_df["baseline_PIM"] = baseline.values
    predictions_df["predicted_PIM"] = predictions

    results.to_csv(
        DATA_DIR / "pim_first_results.csv",
        index=False,
    )

    predictions_df.to_csv(
        DATA_DIR / "pim_first_predictions.csv",
        index=False,
    )

    print("\nResultados guardados.")


if __name__ == "__main__":
    main()
