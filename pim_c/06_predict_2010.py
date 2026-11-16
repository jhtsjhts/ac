
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

# Importar modelos definidos no passo 6
module_path = Path(__file__).parent / "04_two_stage_model.py"

spec = spec_from_file_location("two_stage", module_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

FEATURES = module.FEATURES


def build_future_candidates(data, target_year=2010):

    # Usar apenas informacao anterior a 2010
    history = data[data["year"] < target_year].copy()

    # Jogadores da ultima epoca observada
    candidate_ids = history[
        history["year"] == target_year - 1
    ]["playerID"].unique()

    history = history[
        history["playerID"].isin(candidate_ids)
    ].copy()

    rows = []

    for player_id, group in history.groupby("playerID"):

        group = group.sort_values("year")

        last = group.iloc[-1]
        recent = group.tail(3)

        rows.append({
            "playerID": player_id,
            "year": target_year,
            "pos": last["pos"],

            "prev_PIM": last["PIM"],
            "prev_GP": last["GP"],
            "prev_pim_per_game": last["pim_per_game"],

            "avg_PIM_3": recent["PIM"].mean(),
            "avg_GP_3": recent["GP"].mean(),
            "avg_pim_per_game_3":
                recent["pim_per_game"].mean(),

            "experience": len(group),

            "last_played_year": int(last["year"]),
        })

    return pd.DataFrame(rows)


def main():

    print("\n=== PREVISOES FINAIS PIM 2010 ===")

    dataset = pd.read_csv(
        DATA_DIR / "pim_dataset.csv"
    )

    players = pd.read_csv(
        DATA_DIR / "players_clean.csv"
    )

    train = dataset[
        dataset["year"] <= 2009
    ].copy()

    future = build_future_candidates(
        players,
        target_year=2010,
    )

    print("Treino:", len(train))
    print("Jogadores candidatos:", len(future))

    # MODELO 1 - PARTICIPACAO
    classifier = module.classification_model()

    classifier.fit(
        train[FEATURES],
        train["played_next_year"].astype(int),
    )

    participation_probability = classifier.predict_proba(
        future[FEATURES]
    )[:, 1]

    # MODELO 2 - PIM CONDICIONAL
    played_train = train[
        train["played_next_year"] == True
    ].copy()

    regressor = module.regression_model()

    regressor.fit(
        played_train[FEATURES],
        played_train["target_PIM"],
    )

    conditional_pim = np.maximum(
        0,
        regressor.predict(future[FEATURES]),
    )

    # PREVISAO FINAL
    future["prob_participation"] = (
        participation_probability
    )

    future["pred_pim_if_playing"] = (
        conditional_pim
    )

    future["predicted_PIM"] = (
        participation_probability * conditional_pim
    )

    future["predicted_PIM_rounded"] = (
        np.rint(future["predicted_PIM"]).astype(int)
    )

    # Ordenar jogadores
    future = future.sort_values(
        "predicted_PIM",
        ascending=False,
    )

    # GUARDAR
    output = future[
        [
            "playerID",
            "year",
            "pos",
            "last_played_year",
            "prev_PIM",
            "avg_PIM_3",
            "experience",
            "prob_participation",
            "pred_pim_if_playing",
            "predicted_PIM",
            "predicted_PIM_rounded",
        ]
    ].copy()

    output.to_csv(
        DATA_DIR / "pim_predictions_2010_final.csv",
        index=False,
    )

    print("\n=== TOP 20 PREVISOES ===")
    print(
        output.head(20).round(3).to_string(index=False)
    )

    print("\n=== RESUMO ===")
    print("Total de previsoes:", len(output))
    print(
        "PIM medio previsto:",
        round(output["predicted_PIM"].mean(), 2),
    )
    print(
        "PIM maximo previsto:",
        round(output["predicted_PIM"].max(), 2),
    )

    print("\nFicheiro criado:")
    print(DATA_DIR / "pim_predictions_2010_final.csv")


if __name__ == "__main__":
    main()
