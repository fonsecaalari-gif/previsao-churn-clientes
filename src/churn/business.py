"""Tradução das previsões do modelo em dinheiro.

A pergunta que a área de negócio faz não é "qual o ROC-AUC?", e sim:
*"se eu usar esse modelo para decidir quem recebe uma oferta de retenção,
quanto eu ganho?"*. Este módulo responde isso e escolhe o limiar de decisão que
maximiza o lucro, em vez de usar o 0,5 padrão.
"""

from __future__ import annotations

import numpy as np

from churn.config import (
    CUSTO_ACAO_RETENCAO,
    MESES_RECEITA_PRESERVADA,
    TAXA_SUCESSO_RETENCAO,
)


def lucro_campanha(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mensalidade: np.ndarray,
    custo_acao: float = CUSTO_ACAO_RETENCAO,
    taxa_sucesso: float = TAXA_SUCESSO_RETENCAO,
    meses: int = MESES_RECEITA_PRESERVADA,
) -> float:
    """Lucro líquido de abordar todos os clientes com ``y_pred == 1``.

    - Todo cliente abordado custa ``custo_acao``.
    - Se ele realmente iria cancelar (verdadeiro positivo), há ``taxa_sucesso``
      de chance de retê-lo, preservando ``meses`` de mensalidade.
    - Abordar quem não iria cancelar (falso positivo) é só custo.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(bool)
    mensalidade = np.asarray(mensalidade, dtype=float)

    receita_salva = (taxa_sucesso * meses * mensalidade[y_pred & (y_true == 1)]).sum()
    custo = custo_acao * y_pred.sum()
    return float(receita_salva - custo)


def curva_lucro(
    y_true: np.ndarray,
    proba: np.ndarray,
    mensalidade: np.ndarray,
    limiares: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula o lucro da campanha para cada limiar de probabilidade."""
    if limiares is None:
        limiares = np.round(np.arange(0.05, 0.951, 0.01), 2)
    lucros = np.array(
        [lucro_campanha(y_true, proba >= t, mensalidade) for t in limiares]
    )
    return limiares, lucros


def melhor_limiar(y_true, proba, mensalidade) -> tuple[float, float]:
    """Retorna (limiar, lucro) que maximizam o lucro."""
    limiares, lucros = curva_lucro(y_true, proba, mensalidade)
    i = int(np.argmax(lucros))
    return float(limiares[i]), float(lucros[i])
