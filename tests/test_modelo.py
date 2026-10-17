"""Testes do pipeline completo, com um treino rápido em amostra pequena."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from churn.data import load_dataset
from churn.train import construir_modelos


def test_pipeline_aprende_algo_util():
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    modelo = construir_modelos()["Regressão Logística"].fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_te)[:, 1]
    assert roc_auc_score(y_te, proba) > 0.80  # bem acima do acaso (0,50)


def test_pipeline_recebe_dados_brutos_e_retorna_probabilidade():
    X, y = load_dataset()
    modelo = construir_modelos()["XGBoost"].set_params(modelo__n_estimators=30).fit(X, y)
    proba = modelo.predict_proba(X.head(5))[:, 1]
    assert proba.shape == (5,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_contrato_mensal_tem_mais_risco_que_dois_anos():
    """Teste de sanidade: o modelo deve refletir o padrão mais forte dos dados."""
    X, y = load_dataset()
    modelo = construir_modelos()["Regressão Logística"].fit(X, y)
    cliente = X.iloc[[0]].copy()
    mensal = cliente.assign(Contract="Month-to-month")
    dois_anos = cliente.assign(Contract="Two year")
    assert modelo.predict_proba(mensal)[0, 1] > modelo.predict_proba(dois_anos)[0, 1]
