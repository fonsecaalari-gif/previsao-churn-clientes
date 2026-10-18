"""Engenharia de atributos e pré-processamento.

Tudo fica dentro de um ``sklearn.Pipeline``. Assim, o mesmo objeto salvo em disco
recebe os dados *brutos* (como no CSV) e já devolve a previsão. Assim o app e a
produção aplicam as mesmas transformações do treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SERVICOS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERICAS_ORIGINAIS = ["tenure", "MonthlyCharges", "TotalCharges"]
NUMERICAS_NOVAS = ["num_servicos", "gasto_medio_mensal", "diferenca_preco"]
BINARIAS_NOVAS = ["pagamento_automatico", "sem_suporte_protecao", "cliente_novo"]
CATEGORICAS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "faixa_tempo_casa",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos com significado de negócio a partir das colunas originais."""
    out = df.copy()

    # Quantos serviços o cliente contratou: mais serviços = mais "amarrado" à empresa.
    out["num_servicos"] = (out[SERVICOS] == "Yes").sum(axis=1)

    # Gasto médio real por mês. Para tenure 0 usamos a mensalidade atual.
    meses = out["tenure"].replace(0, np.nan)
    out["gasto_medio_mensal"] = (out["TotalCharges"] / meses).fillna(out["MonthlyCharges"])

    # Mensalidade atual acima da média histórica indica reajuste recente.
    out["diferenca_preco"] = out["MonthlyCharges"] - out["gasto_medio_mensal"]

    out["pagamento_automatico"] = out["PaymentMethod"].str.contains("automatic").astype(int)

    # Tem internet mas nenhum serviço de segurança/suporte.
    protecao = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    out["sem_suporte_protecao"] = (
        (out["InternetService"] != "No") & (out[protecao] != "Yes").all(axis=1)
    ).astype(int)

    out["cliente_novo"] = (out["tenure"] <= 6).astype(int)

    out["faixa_tempo_casa"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)

    return out


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Codifica categóricas (one-hot) e, se pedido, padroniza as numéricas."""
    numericas = NUMERICAS_ORIGINAIS + NUMERICAS_NOVAS
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_step, numericas),
            ("bin", "passthrough", BINARIAS_NOVAS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAS,
            ),
        ],
        verbose_feature_names_out=False,
    )


def feature_engineering_step() -> FunctionTransformer:
    """Envolve ``add_features`` para poder entrar num Pipeline."""
    return FunctionTransformer(add_features, validate=False)
