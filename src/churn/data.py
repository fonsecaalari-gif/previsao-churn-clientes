"""Carregamento e limpeza dos dados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import DATA_RAW, ID_COL, TARGET


def load_raw(path: str | Path = DATA_RAW) -> pd.DataFrame:
    """Lê o CSV original sem nenhuma transformação."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza mínima necessária e devolve uma cópia.

    - ``TotalCharges`` vem como texto e tem 11 valores em branco. Todos são
      clientes com ``tenure == 0`` (acabaram de entrar), então o total gasto é 0.
    - ``SeniorCitizen`` vem como 0/1; padronizamos para "Yes"/"No" como as demais.
    - O alvo ``Churn`` vira 1 (cancelou) / 0 (ficou).
    """
    out = df.copy()

    total = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    out["TotalCharges"] = total.fillna(0.0)

    if pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({1: "Yes", 0: "No"})

    if TARGET in out.columns and not pd.api.types.is_numeric_dtype(out[TARGET]):
        out[TARGET] = (out[TARGET] == "Yes").astype(int)

    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis explicativas (X) e alvo (y), descartando o ID."""
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]
    return X, y


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Atalho: lê, limpa e separa X/y."""
    return split_xy(clean(load_raw()))
