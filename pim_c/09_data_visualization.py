
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "pim_c" / "data" / "players_clean.csv"
)

PLOTS_DIR = BASE_DIR / "pim_c" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# CARREGAR E PREPARAR DADOS
df = pd.read_csv(DATA_FILE)

# Usar epocas comparaveis com o dataset preditivo
# Excluir o ano da temporada cancelada
df = df[
    (df["year"] >= 1968)
    & (df["year"] <= 2009)
    & (df["year"] != 2004)
].copy()

df = df.dropna(subset=["PIM", "GP"])
df = df[df["GP"] > 0].copy()

print("\n=== DATA UNDERSTANDING - PIM ===")
print("Registos utilizados:", len(df))


# GRAFICO 1 - DISTRIBUICAO DOS PIM
fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    df["PIM"],
    bins=50,
    color="#2563EB",
    edgecolor="white",
    alpha=0.9,
)

mean_pim = df["PIM"].mean()
median_pim = df["PIM"].median()

ax.axvline(
    mean_pim,
    color="#DC2626",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {mean_pim:.1f}",
)

ax.axvline(
    median_pim,
    color="#16A34A",
    linestyle="--",
    linewidth=2,
    label=f"Median: {median_pim:.1f}",
)

ax.set_title(
    "Distribution of NHL Player Penalty Minutes",
    fontsize=16,
    fontweight="bold",
)

ax.set_xlabel("Penalty Minutes (PIM)")
ax.set_ylabel("Number of Player-Seasons")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "01_pim_distribution.png",
    dpi=200,
)

plt.close(fig)

# GRAFICO 2 - PIM POR POSICAO
# Manter as cinco posicoes principais
positions = ["C", "D", "L", "R", "G"]

position_data = df[
    df["pos"].isin(positions)
].copy()

summary = (
    position_data.groupby("pos")["PIM"]
    .agg(["mean", "median", "count"])
    .reindex(positions)
)

labels = {
    "C": "Center",
    "D": "Defense",
    "L": "Left Wing",
    "R": "Right Wing",
    "G": "Goalie",
}

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    [labels[p] for p in positions],
    summary["mean"],
    color=[
        "#2563EB",
        "#0F766E",
        "#7C3AED",
        "#EA580C",
        "#64748B",
    ],
)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

ax.set_title(
    "Average Penalty Minutes by Player Position",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel("Average PIM per Player-Season")
ax.set_ylim(0, summary["mean"].max() * 1.2)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "02_pim_by_position.png",
    dpi=200,
)

plt.close(fig)

# RESULTADOS
print("\n=== DISTRIBUICAO PIM ===")
print(f"Media: {mean_pim:.2f}")
print(f"Mediana: {median_pim:.2f}")

print("\n=== PIM POR POSICAO ===")
print(summary.round(2).to_string())

print("\nGraficos criados:")
print(PLOTS_DIR / "01_pim_distribution.png")
print(PLOTS_DIR / "02_pim_by_position.png")
