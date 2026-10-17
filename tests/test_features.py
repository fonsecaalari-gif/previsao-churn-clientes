import numpy as np
import pandas as pd
import pytest

from churn.data import clean, load_raw
from churn.features import add_features, build_preprocessor


def _cliente(**kwargs) -> pd.DataFrame:
    base = {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
        "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0, "TotalCharges": 800.0,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_num_servicos_conta_apenas_yes():
    f = add_features(_cliente())
    # PhoneService, StreamingTV, StreamingMovies
    assert f.loc[0, "num_servicos"] == 3


def test_gasto_medio_com_tenure_zero_usa_mensalidade():
    f = add_features(_cliente(tenure=0, TotalCharges=0.0, MonthlyCharges=70.0))
    assert f.loc[0, "gasto_medio_mensal"] == 70.0
    assert np.isfinite(f.loc[0, "diferenca_preco"])


def test_flags_binarias():
    f = add_features(_cliente(PaymentMethod="Credit card (automatic)", tenure=3))
    assert f.loc[0, "pagamento_automatico"] == 1
    assert f.loc[0, "cliente_novo"] == 1
    assert f.loc[0, "sem_suporte_protecao"] == 1
    assert f.loc[0, "faixa_tempo_casa"] == "0-6m"


def test_preprocessador_sem_nulos_no_dataset_inteiro():
    X = add_features(clean(load_raw()).drop(columns=["Churn", "customerID"]))
    matriz = build_preprocessor().fit_transform(X)
    assert not np.isnan(matriz).any()
    assert matriz.shape[0] == 7043


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_categoria_desconhecida_nao_quebra():
    X = add_features(clean(load_raw()).drop(columns=["Churn", "customerID"]))
    prep = build_preprocessor().fit(X)
    novo = add_features(_cliente(PaymentMethod="Pix (automatic)"))
    assert prep.transform(novo).shape[1] == prep.transform(X.head(1)).shape[1]
