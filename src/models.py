"""Modelos de regressão para a percentagem de pontos relativa.

Imputação e escalamento ficam dentro de cada Pipeline, para serem ajustados
apenas com os dados de treino de cada divisão temporal (sem fuga de informação).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

ENSEMBLE = "ensemble"


def make_models():
    return {
        # Linear: precisa de imputação e escalamento.
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=10.0)),
        # Árvores: não precisam de escalamento.
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=400, min_samples_leaf=10,
                                                             max_features=0.4, n_jobs=-1,
                                                             random_state=RANDOM_STATE)),
        # Trata valores em falta nativamente.
        "hist_gb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                                 min_samples_leaf=20, l2_regularization=1.0,
                                                 random_state=RANDOM_STATE),
    }


def fit_predict_all(train_X, train_y, test_X):
    """Treina todos os modelos e devolve {nome: previsões}, incluindo o ensemble."""
    preds = {}
    for name, model in make_models().items():
        model.fit(train_X, train_y)
        preds[name] = model.predict(test_X)
    preds[ENSEMBLE] = np.mean(list(preds.values()), axis=0)
    return preds


def fit_predict(name, train_X, train_y, test_X):
    """Treina só o modelo escolhido (ou todos, se for o ensemble)."""
    if name == ENSEMBLE:
        return fit_predict_all(train_X, train_y, test_X)[ENSEMBLE]
    model = make_models()[name]
    model.fit(train_X, train_y)
    return model.predict(test_X)
