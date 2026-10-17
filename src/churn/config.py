"""Configurações centrais do projeto (caminhos, semente e premissas de negócio)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"
ID_COL = "customerID"

# ---------------------------------------------------------------------------
# Premissas de negócio (usadas para escolher o limiar de decisão do modelo)
# ---------------------------------------------------------------------------
# Custo de abordar um cliente com uma oferta de retenção (desconto, ligação etc.).
CUSTO_ACAO_RETENCAO = 50.0
# Probabilidade de a ação de retenção funcionar em quem realmente iria cancelar.
TAXA_SUCESSO_RETENCAO = 0.30
# Horizonte (em meses) de receita preservada quando um cliente é retido.
MESES_RECEITA_PRESERVADA = 12
