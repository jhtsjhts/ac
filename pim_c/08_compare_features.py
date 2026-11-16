
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

INPUT_FILE = DATA_DIR / "pim_dataset_extended.csv"

ORIGINAL_NUMERIC = [
    "prev_PIM",
    "prev_GP",
    "prev_pim_per_game",
    "avg_PIM_3",
    "avg_GP_3",
    "avg_pim_per_game_3",
    "experience",
]

EXTENDED_NUMERIC = ORIGINAL_NUMERIC + [
    "age",
    "height",
    "weight",
]


def make_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
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
                ]),
                ["pos"],
            ),
        ]
    )


def create_classifier(numeric_features):
    return Pipeline([
        ("prep", make_preprocessor(numeric_features)),
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


def create_regressor(numeric_features):
    return Pipeline([
        ("prep", make_preprocessor(numeric_features)),
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


def evaluate_configuration(train, validation, numeric_features, name):

    features = numeric_features + ["pos"]

    # Classificador de participacao
    classifier = create_classifier(numeric_features)

    classifier.fit(
        train[features],
        train["played_next_year"].astype(int),
    )

    probabilities = classifier.predict_proba(
        validation[features]
    )[:, 1]

    # Regressor de PIM, apenas para quem participou
    played_train = train[
        train["played_next_year"] == True
    ]

    regressor = create_regressor(numeric_features)

    regressor.fit(
        played_train[features],
        played_train["target_PIM"],
    )

    conditional = np.maximum(
        0,
        regressor.predict(validation[features]),
    )

    predictions = probabilities * conditional

    y_true = validation["target_PIM"]

    return {
        "Modelo": name,
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "AUC": roc_auc_score(
            validation["played_next_year"].astype(int),
            probabilities,
        ),
    }


def main():

    df = pd.read_csv(INPUT_FILE)

    train = df[df["year"] <= 2003].copy()

    validation = df[
        df["year"].isin([2005, 2006])
    ].copy()

    print("\n=== COMPARACAO DE FEATURES ===")

    print("Treino:", len(train))
    print("Validacao:", len(validation))

    results = []

    print("\nA treinar modelo original...")

    results.append(
        evaluate_configuration(
            train,
            validation,
            ORIGINAL_NUMERIC,
            "Original",
        )
    )

    print("A treinar modelo com idade, altura e peso...")

    results.append(
        evaluate_configuration(
            train,
            validation,
            EXTENDED_NUMERIC,
            "Com_idade_altura_peso",
        )
    )

    results_df = pd.DataFrame(results)

    print("\n=== RESULTADOS ===")
    print(results_df.round(3).to_string(index=False))

    results_df.to_csv(
        DATA_DIR / "pim_feature_comparison.csv",
        index=False,
    )

    print("\nResultados guardados.")


if __name__ == "__main__":
    main()
