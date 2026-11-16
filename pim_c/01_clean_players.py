
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "pim_c" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BASE_DIR / "raw_data" / "Scoring.csv"
OUTPUT_FILE = OUTPUT_DIR / "players_clean.csv"


def main():
    print("\n=== PIM C - LIMPEZA E VALIDACAO ===")

    # 1. Carregar os dados originais
    df = pd.read_csv(INPUT_FILE)

    # 2. Filtrar a NHL
    df = df[df["lgID"] == "NHL"].copy()

    print(f"Registos originais NHL: {len(df)}")

    # 3. Normalizar identificadores
    df["playerID"] = df["playerID"].astype("string").str.strip()
    df["pos"] = df["pos"].astype("string").str.strip()

    # 4. Converter valores numericos
    for col in ["year", "stint", "GP", "PIM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Verificar identificadores e duplicados
    invalid_ids = df["playerID"].isna().sum()
    duplicates = df.duplicated().sum()

    print(f"Identificadores em falta: {invalid_ids}")
    print(f"Duplicados exatos: {duplicates}")

    if invalid_ids > 0:
        raise ValueError("Existem jogadores sem identificador.")

    if duplicates > 0:
        raise ValueError(
            "Existem duplicados exatos. Investigar antes de continuar."
        )

    # 6. Verificar valores impossiveis
    invalid_gp = (df["GP"].notna() & (df["GP"] <= 0)).sum()
    invalid_pim = (df["PIM"].notna() & (df["PIM"] < 0)).sum()

    print(f"GP invalidos: {invalid_gp}")
    print(f"PIM negativos: {invalid_pim}")

    if invalid_gp or invalid_pim:
        raise ValueError(
            "Foram encontrados valores numericos invalidos."
        )

    # 7. Identificar valores ausentes
    df["incomplete_record"] = (
        df[["GP", "PIM"]].isna().any(axis=1)
    )

    print(
        "Registos incompletos:",
        int(df["incomplete_record"].sum())
    )

    # 8. Ordenar os registos por jogador, epoca e stint
    df = df.sort_values(
        ["playerID", "year", "stint"],
        na_position="first",
    )

    # 9. Agregar as estatisticas por jogador e epoca
    # Apenas calcular totais quando TODAS as linhas desse jogador/epoca possuem valores conhecidos
    grouped = df.groupby(
        ["playerID", "year"],
        as_index=False,
    )

    players = grouped.agg(
        pos=("pos", "last"),
        team_records=("tmID", "size"),
        GP=(
            "GP",
            lambda x: x.sum(min_count=len(x)),
        ),
        PIM=(
            "PIM",
            lambda x: x.sum(min_count=len(x)),
        ),
        incomplete=(
            "incomplete_record",
            "any",
        ),
    )

    # 10. Estatistica por jogo
    players["pim_per_game"] = (
        players["PIM"] / players["GP"]
    )

    # 11. Validar unicidade
    duplicates_final = players.duplicated(
        ["playerID", "year"]
    ).sum()

    if duplicates_final:
        raise ValueError("Duplicados jogador-epoca.")

    # 12. Guardar os dados
    players = players.sort_values(
        ["playerID", "year"]
    ).reset_index(drop=True)

    players.to_csv(OUTPUT_FILE, index=False)

    # 13. Resumo
    print("\n=== RESULTADOS DA LIMPEZA ===")
    print(f"Jogadores distintos: {players['playerID'].nunique()}")
    print(f"Registos jogador-epoca: {len(players)}")
    print(
        "Jogador-epoca incompletos:",
        int(players["incomplete"].sum()),
    )
    print(
        "Jogador-epoca com varias equipas:",
        int((players["team_records"] > 1).sum()),
    )
    print(
        "PIM disponiveis:",
        int(players["PIM"].notna().sum()),
    )

    print("\n=== DISTRIBUICAO DOS PIM ===")
    print(players["PIM"].describe().round(2))

    print("\n=== VALIDACAO ===")
    print("Duplicados finais:", duplicates_final)
    print("PIM negativos:", int((players["PIM"] < 0).sum()))
    print("GP invalidos:", int((players["GP"] <= 0).sum()))

    print(f"\nFicheiro criado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
