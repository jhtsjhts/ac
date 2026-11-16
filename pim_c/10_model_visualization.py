
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pim_c" / "data"
PLOTS_DIR = BASE_DIR / "pim_c" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = DATA_DIR / "pim_final_comparison.csv"
PREDICTIONS_FILE = DATA_DIR / "pim_final_two_stage_predictions.csv"

# CARREGAR DADOS
results = pd.read_csv(RESULTS_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)

print("\n=== VISUALIZACAO DOS MODELOS PIM ===")
print("Modelos:", len(results))
print("Previsoes de teste:", len(predictions))

# GRAFICO 3 - COMPARACAO DOS MODELOS
model_labels = {
    "Baseline": "Baseline",
    "RandomForest direto": "Random Forest",
    "Modelo duas fases": "Two-stage Model",
}

results["Label"] = (
    results["Modelo"].map(model_labels)
    .fillna(results["Modelo"])
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics = [
    ("MAE", "Mean Absolute Error (PIM)"),
    ("RMSE", "Root Mean Squared Error (PIM)"),
]

for ax, (metric, title) in zip(axes, metrics):

    bars = ax.bar(
        results["Label"],
        results[metric],
        color=["#94A3B8", "#2563EB", "#0F766E"],
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Error (PIM)")
    ax.grid(axis="y", alpha=0.2)

    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f}",
            ha="center",
            fontweight="bold",
        )

fig.suptitle(
    "PIM Prediction - Model Comparison",
    fontsize=17,
    fontweight="bold",
)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "03_model_comparison.png",
    dpi=200,
)

plt.close(fig)

# GRAFICO 4 - REAL VS PREVISTO
actual = predictions["target_PIM"]
predicted = predictions["pred_two_stage"]

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    actual,
    predicted,
    alpha=0.30,
    s=18,
    color="#2563EB",
)

max_value = max(actual.max(), predicted.max())

ax.plot(
    [0, max_value],
    [0, max_value],
    color="#DC2626",
    linestyle="--",
    linewidth=2,
    label="Perfect prediction",
)

ax.set_title(
    "Actual vs Predicted Penalty Minutes",
    fontsize=16,
    fontweight="bold",
)

ax.set_xlabel("Actual PIM")
ax.set_ylabel("Predicted PIM")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "04_actual_vs_predicted.png",
    dpi=200,
)

plt.close(fig)

# GRAFICO 5 - DISTRIBUICAO DOS ERROS
errors = actual - predicted

fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    errors,
    bins=60,
    color="#7C3AED",
    edgecolor="white",
    alpha=0.90,
)

ax.axvline(
    0,
    color="#DC2626",
    linestyle="--",
    linewidth=2,
    label="Zero error",
)

ax.set_title(
    "Distribution of PIM Prediction Errors",
    fontsize=16,
    fontweight="bold",
)

ax.set_xlabel("Actual PIM - Predicted PIM")
ax.set_ylabel("Number of Player-Seasons")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    PLOTS_DIR / "05_prediction_errors.png",
    dpi=200,
)

plt.close(fig)

# RESULTADOS
print("\nGraficos criados:")
print("03_model_comparison.png")
print("04_actual_vs_predicted.png")
print("05_prediction_errors.png")

print("\n=== RESUMO DOS ERROS ===")
print(f"Erro medio assinado: {errors.mean():.3f}")
print(f"Erro absoluto medio: {errors.abs().mean():.3f}")
print(f"Maior subestimacao: {errors.max():.3f}")
print(f"Maior sobrestimacao: {errors.min():.3f}")

print("\nConcluido.")
