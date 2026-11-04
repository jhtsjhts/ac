"""Caminhos e constantes partilhados por todo o projeto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "hockey_development_data" / "hockey_development_data"
OUT_DIR = BASE_DIR / "outputs"

LEAGUE = "NHL"
RANDOM_STATE = 42

# Tarefa (a) — ranking
FIRST_TRAIN_SEASON = 1970      # época moderna (pós-expansão de 1967)
FIRST_EVAL_SEASON = 1995       # épocas usadas na validação walk-forward
N_LAGS = 3                     # quantas épocas anteriores usar
GAMES_PER_SEASON = 82
