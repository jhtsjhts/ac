
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

df = pd.read_csv(DATA_DIR / "pim_dataset.csv")

NUMERIC = [
    "prev_PIM",
    "prev_GP",
    "prev_pim_per_game",
    "avg_PIM_3",
    "avg_GP_3",
    "avg_pim_per_game_3",
    "experience",
]

FEATURES = NUMERIC + ["pos"]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC,
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
                ["pos"],
            ),
        ]
    )


def regression_model():
    return Pipeline([
        ("prep", make_preprocessor()),
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
    ])


def classification_model():
    return Pipeline([
        ("prep", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def evaluate(y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    return mae, rmse


def main():
    train = df[df["year"] <= 2003].copy()
    validation = df[df["year"].isin([2005, 2006])].copy()

    print("\n=== MODELO EM DUAS FASES ===")
    print("Treino:", len(train))
    print("Validacao:", len(validation))

    # Modelo direto de regressao
    direct = regression_model()
    direct.fit(train[FEATURES], train["target_PIM"])

    pred_direct = np.maximum(
        0,
        direct.predict(validation[FEATURES]),
    )

    # Modelo 1: probabilidade de participacao
    classifier = classification_model()
    classifier.fit(
        train[FEATURES],
        train["played_next_year"].astype(int),
    )

    participation_probability = (
        classifier.predict_proba(
            validation[FEATURES]
        )[:, 1]
    )

    # Modelo 2: PIM caso participe
    played_train = train[
        train["played_next_year"] == True
    ].copy()

    conditional = regression_model()

    conditional.fit(
        played_train[FEATURES],
        played_train["target_PIM"],
    )

    conditional_pim = np.maximum(
        0,
        conditional.predict(
            validation[FEATURES]
        ),
    )

    pred_two_stage = (
        participation_probability * conditional_pim
    )

    baseline = (
        validation["avg_PIM_3"]
        .fillna(validation["prev_PIM"])
        .fillna(0)
        .clip(lower=0)
        .to_numpy()
    )

    y = validation["target_PIM"].to_numpy()

    print("\n=== RESULTADOS VALIDACAO ===")

    for name, pred in [
        ("Baseline", baseline),
        ("RandomForest direto", pred_direct),
        ("Modelo duas fases", pred_two_stage),
    ]:
        mae, rmse = evaluate(y, pred)
        print(
            f"{name}: "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}"
        )

    auc = roc_auc_score(
        validation["played_next_year"].astype(int),
        participation_probability,
    )

    print(f"\nAUC participacao: {auc:.3f}")

    # Guardar resultados
    result = validation[
        [
            "playerID",
            "year",
            "target_PIM",
            "played_next_year",
        ]
    ].copy()

    result["pred_baseline"] = baseline
    result["pred_direct"] = pred_direct
    result["prob_participation"] = participation_probability
    result["pred_conditional"] = conditional_pim
    result["pred_two_stage"] = pred_two_stage

    result.to_csv(
        DATA_DIR / "pim_two_stage_validation.csv",
        index=False,
    )

    print("\nValidacao guardada com sucesso.")


if __name__ == "__main__":
    main()
