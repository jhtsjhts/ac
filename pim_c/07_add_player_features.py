
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"

MASTER_FILE = BASE_DIR / "raw_data" / "Master.csv"
DATASET_FILE = DATA_DIR / "pim_dataset.csv"
OUTPUT_FILE = DATA_DIR / "pim_dataset_extended.csv"


def main():
    print("\n=== FEATURES BIOGRAFICAS ===")

    # Carregar dados
    df = pd.read_csv(DATASET_FILE)
    master = pd.read_csv(MASTER_FILE)

    # Selecionar apenas atributos conhecidos antes do inicio da epoca
    master = master[
        [
            "playerID",
            "birthYear",
            "height",
            "weight",
        ]
    ].copy()

    # Normalizar identificadores
    df["playerID"] = df["playerID"].astype(str).str.strip()
    master["playerID"] = (
        master["playerID"].astype("string").str.strip()
    )

    master = master.dropna(subset=["playerID"])

    # Validar unicidade da tabela Master
    duplicates = master["playerID"].duplicated().sum()

    print("Duplicados na Master:", duplicates)

    if duplicates:
        raise ValueError(
            "Existem playerID duplicados na Master."
        )

    # Converter colunas numericas
    for col in ["birthYear", "height", "weight"]:
        master[col] = pd.to_numeric(
            master[col],
            errors="coerce",
        )

    # Integrar com o dataset preditivo
    df = df.merge(
        master,
        on="playerID",
        how="left",
        validate="many_to_one",
    )

    # Idade aproximada no inicio da epoca
    # Apenas utiliza ano de nascimento
    df["age"] = df["year"] - df["birthYear"]

    # Verificar valores impossiveis
    invalid_age = (
        df["age"].notna()
        & ~df["age"].between(15, 65)
    )

    print("Idades fora de 15-65:", invalid_age.sum())

    # Idades suspeitas ficam em falta
    df.loc[invalid_age, "age"] = pd.NA

    # Validar que a integracao nao criou linhas
    if len(df) != len(pd.read_csv(DATASET_FILE)):
        raise ValueError(
            "A integracao alterou o numero de registos."
        )

    # Guardar o dataset experimental
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== RESULTADOS ===")
    print("Total de registos:", len(df))

    print("\nValores em falta:")
    print(
        df[["age", "height", "weight"]]
        .isna()
        .sum()
    )

    print("\nEstatisticas:")
    print(
        df[["age", "height", "weight"]]
        .describe()
        .round(2)
    )

    print("\nDataset criado:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
