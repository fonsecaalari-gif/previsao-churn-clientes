# 📉 Previsão de Churn de Clientes — do dado ao dinheiro

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-189FDD)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
[![testes](https://github.com/fonsecaalari-gif/previsao-churn-clientes/actions/workflows/ci.yml/badge.svg)](https://github.com/fonsecaalari-gif/previsao-churn-clientes/actions)

> Modelo de Machine Learning que identifica clientes com risco de cancelar e decide **quem deve receber uma oferta de retenção para maximizar o lucro**, e não apenas uma métrica técnica.

**🚀 [Testar o app ao vivo](https://previsao-churn-larissa.streamlit.app)** · **📓 [Análise exploratória](notebooks/01_analise_exploratoria.ipynb)** · **🤖 [Modelagem](notebooks/02_modelagem.ipynb)**

---

## 🎯 Resultado em uma frase

Usando o modelo para escolher quem abordar, a campanha de retenção gera **US$ 52 mil de lucro** em 1.409 clientes de teste: **31% a mais** do que o mesmo modelo com o limiar padrão de 0,5 e **quase 4x mais** do que abordar o mesmo número de clientes ao acaso.

| Estratégia (conjunto de teste, 1.409 clientes) | Lucro da campanha |
|---|---:|
| Não fazer nada | US$ 0 |
| Abordar clientes ao acaso (mesmo volume do modelo) | US$ 13.319 |
| Abordar 100% da base | US$ 27.524 |
| Modelo com limiar padrão (0,50) | US$ 39.814 |
| **Modelo com limiar otimizado pelo lucro (0,20)** | **US$ 52.111** |

---

## 🧩 O problema

Uma operadora de telecom perde **26,5% dos clientes**, e esses clientes representam **30,5% da receita mensal**. Reter um cliente é mais barato que conquistar um novo, mas oferecer desconto para todo mundo custa caro. A pergunta de negócio é:

> **Quais clientes devemos abordar com uma oferta de retenção para ganhar o máximo de dinheiro?**

## 🔍 Principais descobertas

<table>
<tr>
<td><img src="reports/figures/01_churn_por_contrato.png" width="100%"></td>
<td><img src="reports/figures/02_churn_por_tempo_de_casa.png" width="100%"></td>
</tr>
</table>

- **Contrato mensal** tem 42,7% de churn, contra 2,8% no contrato de 2 anos (15x mais).
- **55% dos cancelamentos acontecem no primeiro ano.** Nos 6 primeiros meses, o churn é de 52,9%.
- **Pagamento por cheque eletrônico** cancela 3x mais que pagamento automático (45% x 15%).
- Clientes com internet **sem nenhum serviço de suporte ou segurança** cancelam 56,7% das vezes.
- O segmento **contrato mensal + fibra óptica + até 12 meses** tem **70% de churn** e concentra 13% da base.
- **Gênero não influencia** o churn (26,9% x 26,2%), então o modelo não depende de variáveis sensíveis.

## 🤖 Modelagem

**Abordagem para evitar vazamento de dados:** teste separado logo no início (20%, estratificado), todo pré-processamento dentro de um `sklearn.Pipeline` e limiar de decisão escolhido com previsões *out-of-fold* do treino. O teste é usado uma única vez.

| Modelo (validação cruzada, 5 folds) | ROC-AUC | PR-AUC |
|---|---:|---:|
| Baseline (sem modelo) | 0,500 | 0,265 |
| Random Forest | 0,844 ± 0,010 | 0,657 ± 0,024 |
| Regressão Logística | 0,847 ± 0,012 | 0,660 ± 0,015 |
| **XGBoost** | **0,847 ± 0,010** | **0,664 ± 0,022** |

**Leitura honesta:** os três modelos empatam dentro da margem de erro. Escolhi o XGBoost pela maior PR-AUC média, mas a Regressão Logística seria uma alternativa igualmente boa e mais fácil de explicar para o negócio.

<img src="reports/figures/05_curvas_roc_pr.png" width="100%">

### De probabilidade para decisão: o limiar que dá mais lucro

O padrão é classificar como "vai cancelar" quem tem probabilidade ≥ 0,50. Mas **deixar de abordar um cliente que vai sair custa muito mais do que abordar alguém que ficaria**. Por isso o limiar é escolhido pelo lucro, com premissas explícitas e ajustáveis em [`config.py`](src/churn/config.py):

| Premissa | Valor |
|---|---|
| Custo da oferta por cliente abordado | US$ 50 |
| Chance de a oferta reter quem iria sair | 30% |
| Receita preservada por cliente retido | 12 mensalidades |

<table>
<tr>
<td width="60%"><img src="reports/figures/07_curva_lucro.png" width="100%"></td>
<td width="40%"><img src="reports/figures/06_matriz_confusao.png" width="100%"></td>
</tr>
</table>

No limiar de 0,20, o modelo encontra **84,5% dos clientes que iriam cancelar** abordando 47% da base (ROC-AUC 0,846 e PR-AUC 0,666 no teste).

### O que o modelo aprendeu

<img src="reports/figures/08_importancia.png" width="65%">

Tempo de casa e tipo de contrato dominam a previsão, confirmando a análise exploratória.

## 💡 Recomendações para o negócio

1. **Campanha de migração de contrato:** oferecer desconto para trocar o plano mensal pelo anual nos primeiros 6 meses.
2. **Incentivo ao pagamento automático:** bônus na fatura para quem trocar o cheque eletrônico por débito ou cartão.
3. **Pacote de boas-vindas para fibra:** suporte técnico e segurança online gratuitos nos primeiros meses.
4. **Usar o modelo mensalmente** para priorizar quem a equipe de retenção deve contatar.

## 🖥️ App interativo

Avalie o risco de qualquer cliente e simule a campanha com suas próprias premissas de custo e taxa de sucesso.

<img src="reports/figures/app_streamlit.png" width="100%">

## 🗂️ Estrutura do projeto

```
previsao-churn-clientes/
├── app/streamlit_app.py        # app interativo (avaliação de cliente + simulador de campanha)
├── data/raw/                   # dataset original, sem alterações
├── models/modelo_churn.joblib  # pipeline treinado + limiar escolhido
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_modelagem.ipynb
├── reports/
│   ├── figures/                # gráficos usados neste README
│   └── metrics.json            # todas as métricas, geradas automaticamente
├── src/churn/
│   ├── config.py               # caminhos e premissas de negócio
│   ├── data.py                 # leitura e limpeza
│   ├── features.py             # engenharia de atributos + pré-processamento
│   ├── business.py             # cálculo de lucro e escolha do limiar
│   ├── plots.py                # gráficos com estilo padronizado
│   └── train.py                # pipeline completo de treino e avaliação
├── tests/                      # 18 testes automatizados (pytest)
└── .github/workflows/ci.yml    # roda testes e treino a cada push
```

## ▶️ Como reproduzir

```bash
git clone https://github.com/fonsecaalari-gif/previsao-churn-clientes.git
cd previsao-churn-clientes
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

pytest -q                          # testes
python -m churn.train              # treina e gera métricas e gráficos (~30 s)
streamlit run app/streamlit_app.py # abre o app
```

## ⚠️ Limitações e próximos passos

- O dataset é um retrato de um único momento: dados de uso, reclamações e atendimento provavelmente melhorariam bastante o modelo.
- As premissas de custo e taxa de sucesso são estimativas; em um cenário real, viriam de um teste A/B.
- Próximos passos: calibração das probabilidades, explicações individuais com SHAP, monitoramento de *drift* e uma API com FastAPI.

## 🛠️ Tecnologias

Python · pandas · scikit-learn · XGBoost · matplotlib · Streamlit · pytest · GitHub Actions

---

**Dados:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (Apache 2.0) · valores em dólares.

**Autora:** Larissa Fonseca · [LinkedIn](https://www.linkedin.com/in/larissa-fonseca-179885170/)
