import numpy as np
import pytest

from churn.business import curva_lucro, lucro_campanha, melhor_limiar


def test_nao_abordar_ninguem_da_lucro_zero():
    y = np.array([1, 0, 1])
    assert lucro_campanha(y, np.zeros(3), np.array([100, 100, 100])) == 0.0


def test_lucro_calculado_a_mao():
    y = np.array([1, 0])
    pred = np.array([1, 1])
    mensal = np.array([100.0, 80.0])
    # receita salva: 0,30 * 12 * 100 = 360 | custo: 2 * 50 = 100
    assert lucro_campanha(y, pred, mensal, custo_acao=50, taxa_sucesso=0.30, meses=12) == pytest.approx(260.0)


def test_falso_positivo_so_gera_custo():
    assert lucro_campanha(np.array([0]), np.array([1]), np.array([100.0]), custo_acao=50) == -50.0


def test_melhor_limiar_separa_classes_perfeitas():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    mensal = np.full(5, 100.0)
    limiar, lucro = melhor_limiar(y, proba, mensal)
    assert 0.3 < limiar <= 0.8
    assert lucro == pytest.approx(2 * (0.3 * 12 * 100 - 50))


def test_curva_tem_um_valor_por_limiar():
    lim, luc = curva_lucro(np.array([0, 1]), np.array([0.2, 0.7]), np.array([50.0, 50.0]))
    assert len(lim) == len(luc)
