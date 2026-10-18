"""App interativo: risco de churn de um cliente + simulador de campanha de retenção.

Rodar localmente:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from churn.business import lucro_campanha  # noqa: E402
from churn.config import MODELS_DIR, RANDOM_STATE, TEST_SIZE  # noqa: E402
from churn.data import load_dataset  # noqa: E402

st.set_page_config(page_title="Previsão de Churn", page_icon=":material/trending_down:", layout="wide")

CAMINHO_MODELO = MODELS_DIR / "modelo_churn.joblib"


@st.cache_resource
def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        st.error("Modelo não encontrado. Rode `python -m churn.train` antes de abrir o app.")
        st.stop()
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def dados_teste():
    X, y = load_dataset()
    _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_te, y_te


artefato = carregar_modelo()
modelo, limiar_padrao = artefato["pipeline"], artefato["limiar"]

st.title("Previsão de churn de clientes")
st.caption(
    f"Modelo: **{artefato['modelo']}** · treinado com 5.634 clientes reais de telecom (dataset IBM Telco). "
    "Código e análise no repositório do GitHub."
)

aba1, aba2 = st.tabs(["Avaliar um cliente", "Simular campanha de retenção"])

# --------------------------------------------------------------------------- aba 1
with aba1:
    sim_nao = {"Sim": "Yes", "Não": "No"}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Perfil")
        idoso = st.radio("Idoso (65+)?", ["Não", "Sim"], horizontal=True)
        parceiro = st.radio("Tem cônjuge?", ["Sim", "Não"], horizontal=True)
        dependentes = st.radio("Tem dependentes?", ["Não", "Sim"], horizontal=True)
        genero = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        tenure = st.slider("Tempo de casa (meses)", 0, 72, 4)
    with c2:
        st.subheader("Contrato e cobrança")
        contrato = st.selectbox("Contrato", ["Mensal", "1 ano", "2 anos"])
        pagamento = st.selectbox(
            "Forma de pagamento",
            ["Cheque eletrônico", "Cheque pelo correio", "Transferência (automática)", "Cartão de crédito (automático)"],
        )
        fatura_digital = st.radio("Fatura digital?", ["Sim", "Não"], horizontal=True)
        mensalidade = st.slider("Mensalidade (US$)", 18.0, 120.0, 85.0, 0.5)
    with c3:
        st.subheader("Serviços")
        telefone = st.radio("Telefone fixo?", ["Sim", "Não"], horizontal=True)
        multiplas = st.radio("Múltiplas linhas?", ["Não", "Sim"], horizontal=True, disabled=telefone == "Não")
        internet = st.selectbox("Internet", ["Fibra óptica", "DSL", "Sem internet"])
        tem_internet = internet != "Sem internet"
        extras = st.multiselect(
            "Serviços adicionais",
            ["Segurança online", "Backup online", "Proteção de aparelho", "Suporte técnico", "Streaming de TV", "Streaming de filmes"],
            default=["Streaming de TV"] if tem_internet else [],
            disabled=not tem_internet,
        )

    def servico(nome):
        if not tem_internet:
            return "No internet service"
        return "Yes" if nome in extras else "No"

    cliente = pd.DataFrame([{
        "gender": "Female" if genero == "Feminino" else "Male",
        "SeniorCitizen": sim_nao[idoso],
        "Partner": sim_nao[parceiro],
        "Dependents": sim_nao[dependentes],
        "tenure": tenure,
        "PhoneService": sim_nao[telefone],
        "MultipleLines": "No phone service" if telefone == "Não" else sim_nao[multiplas],
        "InternetService": {"Fibra óptica": "Fiber optic", "DSL": "DSL", "Sem internet": "No"}[internet],
        "OnlineSecurity": servico("Segurança online"),
        "OnlineBackup": servico("Backup online"),
        "DeviceProtection": servico("Proteção de aparelho"),
        "TechSupport": servico("Suporte técnico"),
        "StreamingTV": servico("Streaming de TV"),
        "StreamingMovies": servico("Streaming de filmes"),
        "Contract": {"Mensal": "Month-to-month", "1 ano": "One year", "2 anos": "Two year"}[contrato],
        "PaperlessBilling": sim_nao[fatura_digital],
        "PaymentMethod": {
            "Cheque eletrônico": "Electronic check",
            "Cheque pelo correio": "Mailed check",
            "Transferência (automática)": "Bank transfer (automatic)",
            "Cartão de crédito (automático)": "Credit card (automatic)",
        }[pagamento],
        "MonthlyCharges": mensalidade,
        "TotalCharges": mensalidade * tenure,
    }])

    proba = float(modelo.predict_proba(cliente)[0, 1])
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Probabilidade de cancelar", f"{proba:.0%}")
    valor_esperado = proba * 0.30 * 12 * mensalidade - 50
    r2.metric("Valor esperado de abordar", f"US$ {valor_esperado:,.0f}")
    if proba >= limiar_padrao:
        r3.error(f"**Abordar com oferta de retenção** (risco acima do limiar de {limiar_padrao:.0%})")
    else:
        r3.success(f"**Não precisa de ação agora** (risco abaixo do limiar de {limiar_padrao:.0%})")
    st.progress(min(proba, 1.0))
    st.caption("Experimente mudar o contrato para **2 anos** ou o pagamento para **automático** e veja o risco cair.")

# --------------------------------------------------------------------------- aba 2
with aba2:
    st.markdown(
        "Ajuste as premissas da campanha e veja qual limiar de risco dá o maior lucro "
        "nos **1.409 clientes do conjunto de teste** (que o modelo nunca viu no treino)."
    )
    p1, p2, p3 = st.columns(3)
    custo = p1.number_input("Custo da oferta por cliente (US$)", 0.0, 500.0, 50.0, 5.0)
    sucesso = p2.slider("Chance de a oferta reter quem ia sair", 0.05, 1.0, 0.30, 0.05)
    meses = p3.slider("Meses de receita preservada", 1, 36, 12)

    X_te, y_te = dados_teste()
    proba_te = modelo.predict_proba(X_te)[:, 1]
    mensal = X_te["MonthlyCharges"].values
    limiares = np.round(np.arange(0.05, 0.951, 0.01), 2)
    lucros = [lucro_campanha(y_te.values, proba_te >= t, mensal, custo, sucesso, meses) for t in limiares]
    i = int(np.argmax(lucros))

    m1, m2, m3 = st.columns(3)
    m1.metric("Melhor limiar", f"{limiares[i]:.2f}")
    m2.metric("Lucro no melhor limiar", f"US$ {lucros[i]:,.0f}")
    m3.metric("Clientes abordados", f"{int((proba_te >= limiares[i]).sum())} de {len(X_te)}")
    st.line_chart(pd.DataFrame({"Lucro da campanha (US$)": lucros}, index=pd.Index(limiares, name="Limiar")))
