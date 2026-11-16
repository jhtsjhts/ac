
from pathlib import Path
import numpy as np
import pandas as pd

# CAMINHOS
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "pim_c" / "data" / "players_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "pim_c" / "data" / "pim_dataset.csv"
)


# CONSTRUIR DADOS DE UMA EPOCA
def build_season(data, target_year, previous_year=None):

    if previous_year is None:
        previous_year = target_year - 1

    # Apenas informacao anterior a epoca prevista
    history = data[
        data["year"] < target_year
    ].copy()

    # Selecionar os jogadores da epoca anterior
    candidates = history[
        history["year"] == previous_year
    ]["playerID"].unique()

    history = history[
        history["playerID"].isin(candidates)
    ].copy()

    rows = []

    for player_id, group in history.groupby("playerID"):

        group = group.sort_values("year")

        last = group.iloc[-1]
        recent = group.tail(3)

        # Historico conhecido antes da nova epoca
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

            "previous_year": int(last["year"]),
        })

    candidates_df = pd.DataFrame(rows)

    if candidates_df.empty:
        return candidates_df

    # VALORES REAIS DA EPOCA PREVISTA
    # Utilizados apenas como target para treino/avaliacao. Nunca como features.
    actual = data[
        data["year"] == target_year
    ][["playerID", "PIM"]].copy()

    actual = actual.rename(
        columns={"PIM": "target_PIM"}
    )

    actual["has_record"] = True

    candidates_df = candidates_df.merge(
        actual,
        on="playerID",
        how="left",
        validate="one_to_one",
    )

    # Jogadores sem registo na epoca seguinte
    # acumulam 0 PIM na NHL
    #
    # Jogadores com registo mas PIM desconhecido
    # continuam com target NaN
    no_record = candidates_df["has_record"].isna()

    candidates_df.loc[
        no_record, "target_PIM"
    ] = 0.0

    candidates_df["played_next_year"] = (
        ~no_record
    )

    candidates_df = candidates_df.drop(
        columns=["has_record"]
    )

    return candidates_df


# EXECUCAO
def main():

    print("\n=== CONSTRUCAO DATASET PIM ===")

    data = pd.read_csv(INPUT_FILE)

    data = data.sort_values(
        ["playerID", "year"]
    )

    datasets = []

    # Comecar em 1968 para evitar combinar diretamente estruturas muito antigas da NHL
    #
    # 2004: epoca cancelada.
    # 2005: nao usar candidatos de 2004, pois nao existiu uma epoca regular nesse ano
    for year in range(1968, 2010):
        # A temporada 2004-05 foi cancelada
        if year == 2004:
            continue
        # Em 2005, usar a ultima temporada disputada (2003)
        # Nao criar dados ficticios para 2004
        previous_year = 2003 if year == 2005 else year - 1

        season = build_season(
            data,
            year,
            previous_year=previous_year,
        )

        if not season.empty:
            datasets.append(season)

            print(
                f"{year}: "
                f"{len(season)} candidatos"
            )

    df = pd.concat(
        datasets,
        ignore_index=True,
    )

    # VALIDACOES
    duplicates = df.duplicated(
        ["playerID", "year"]
    ).sum()

    if duplicates:
        raise ValueError(
            "Foram encontrados jogadores duplicados."
        )

    invalid = (
        df["target_PIM"].notna()
        & (df["target_PIM"] < 0)
    ).sum()

    if invalid:
        raise ValueError(
            "Foram encontrados PIM negativos."
        )

    # Remover apenas exemplos sem target conhecido para treino/avaliacao
    missing_targets = df["target_PIM"].isna().sum()

    df = df.dropna(
        subset=["target_PIM"]
    ).copy()

    # GUARDAR
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== RESULTADOS ===")

    print("Total de exemplos:", len(df))

    print(
        "Jogadores distintos:",
        df["playerID"].nunique(),
    )

    print(
        "Jogadores que participaram:",
        int(df["played_next_year"].sum()),
    )

    print(
        "Jogadores que nao participaram:",
        int((~df["played_next_year"]).sum()),
    )

    print(
        "Targets desconhecidos excluidos:",
        missing_targets,
    )

    print(
        "Duplicados:",
        duplicates,
    )

    print(
        "Primeira epoca:",
        int(df["year"].min()),
    )

    print(
        "Ultima epoca:",
        int(df["year"].max()),
    )

    print("\n=== PIM ===")
    print(df["target_PIM"].describe().round(2))

    print("\n=== VALORES EM FALTA NAS FEATURES ===")

    features = [
        "prev_PIM",
        "prev_GP",
        "prev_pim_per_game",
        "avg_PIM_3",
        "avg_GP_3",
        "avg_pim_per_game_3",
        "experience",
    ]

    print(df[features].isna().sum())

    print("\nDataset criado:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
