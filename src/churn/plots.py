"""Gráficos do projeto, com um estilo visual único e consistente.

Paleta: azul = "ficou", laranja = "cancelou" (par validado para daltonismo).
Os textos usam tons de cinza, nunca a cor da série.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
CINZA_SERIE = "#8f8e88"
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e6e5e0"

CORES_MODELOS = {
    "Regressão Logística": AZUL,
    "Random Forest": LARANJA,
    "XGBoost": VERDE_AGUA,
    "Baseline (sem modelo)": CINZA_SERIE,
}


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": GRADE,
            "axes.labelcolor": TEXTO_2,
            "axes.titlecolor": TEXTO,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXTO_2,
            "ytick.color": TEXTO_2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def br(x: float, casas: int = 2) -> str:
    """Formata número no padrão brasileiro: 1.234,56."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# Tradução das categorias exibidas nos gráficos.
ROTULOS_CATEGORIAS = {
    "Month-to-month": "Mensal",
    "One year": "1 ano",
    "Two year": "2 anos",
    "Bank transfer (automatic)": "Transferência (automática)",
    "Credit card (automatic)": "Cartão de crédito (automático)",
    "Electronic check": "Cheque eletrônico",
    "Mailed check": "Cheque pelo correio",
    "Fiber optic": "Fibra óptica",
    "No": "Não",
    "Yes": "Sim",
    "0-6m": "0 a 6 meses",
    "7-12m": "7 a 12 meses",
    "13-24m": "13 a 24 meses",
    "25-48m": "25 a 48 meses",
    "49m+": "49 meses ou mais",
}

# Nomes amigáveis para as colunas originais do dataset.
NOMES_COLUNAS = {
    "tenure": "Tempo de casa (meses)",
    "Contract": "Tipo de contrato",
    "InternetService": "Tipo de internet",
    "OnlineSecurity": "Segurança online",
    "TechSupport": "Suporte técnico",
    "PaymentMethod": "Forma de pagamento",
    "PaperlessBilling": "Fatura digital",
    "MultipleLines": "Múltiplas linhas",
    "TotalCharges": "Total já pago",
    "MonthlyCharges": "Mensalidade",
    "SeniorCitizen": "Idoso",
    "StreamingMovies": "Streaming de filmes",
    "StreamingTV": "Streaming de TV",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho",
    "PhoneService": "Telefone",
    "Partner": "Tem cônjuge",
    "Dependents": "Tem dependentes",
    "gender": "Gênero",
}


def _salvar(fig, caminho: Path | None):
    if caminho is not None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho)
        plt.close(fig)
    return fig


def taxa_churn_por(df: pd.DataFrame, coluna: str, titulo: str, ordem=None, caminho=None):
    """Barras horizontais com a taxa de cancelamento em cada categoria."""
    aplicar_estilo()
    taxa = df.groupby(coluna, observed=True)["Churn"].mean()
    n = df.groupby(coluna, observed=True).size()
    if ordem is not None:
        taxa, n = taxa.reindex(ordem), n.reindex(ordem)
    media = df["Churn"].mean()

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(taxa) + 1.5))
    y = np.arange(len(taxa))
    ax.barh(y, taxa.values, color=LARANJA, height=0.6, edgecolor=SUPERFICIE, linewidth=2, zorder=2)
    ax.axvline(media, color=TEXTO_2, linestyle="--", linewidth=1, zorder=1)
    ax.text(media, -0.62, f" média geral {media:.0%}", color=TEXTO_2, fontsize=9, va="bottom")
    for yi, (t, ni) in enumerate(zip(taxa.values, n.values)):
        ax.text(t + 0.008, yi, f"{t:.0%}  (n={ni:,})".replace(",", "."), va="center", color=TEXTO, fontsize=9.5,
                zorder=3, bbox=dict(boxstyle="square,pad=0.15", facecolor=SUPERFICIE, edgecolor="none"))
    rotulos = [ROTULOS_CATEGORIAS.get(str(c), str(c)) for c in taxa.index]
    ax.set_yticks(y, rotulos)
    ax.set_ylim(len(taxa) - 0.5, -0.9)
    ax.set_xlim(0, max(taxa.max() * 1.35, media * 1.4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Taxa de cancelamento")
    ax.set_title(titulo)
    return _salvar(fig, caminho)


def distribuicao_por_churn(df: pd.DataFrame, coluna: str, titulo: str, xlabel: str, bins=30, caminho=None):
    """Histogramas sobrepostos (densidade) de quem ficou x quem cancelou."""
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(7.2, 4))
    faixa = np.histogram_bin_edges(df[coluna], bins=bins)
    for valor, cor, nome in [(0, AZUL, "Ficou"), (1, LARANJA, "Cancelou")]:
        ax.hist(
            df.loc[df["Churn"] == valor, coluna], bins=faixa, density=True,
            histtype="step", color=cor, label=nome, linewidth=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidade")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title(titulo)
    return _salvar(fig, caminho)


def curvas_roc_pr(resultados: dict, y_test, caminho=None):
    """ROC e Precisão-Recall de todos os modelos, lado a lado (mesmo eixo de 0 a 1)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    aplicar_estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for nome, proba in resultados.items():
        cor = CORES_MODELOS.get(nome, TEXTO_2)
        estilo = "--" if nome.startswith("Baseline") else "-"
        fpr, tpr, _ = roc_curve(y_test, proba)
        a1.plot(fpr, tpr, color=cor, ls=estilo, label=f"{nome}  (AUC {br(roc_auc_score(y_test, proba), 3)})")
        p, r, _ = precision_recall_curve(y_test, proba)
        a2.plot(r, p, color=cor, ls=estilo, label=f"{nome}  (AP {br(average_precision_score(y_test, proba), 3)})")
    a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos (recall)", xlim=(0, 1), ylim=(0, 1.01))
    a1.set_title("Curva ROC (conjunto de teste)")
    a2.set(xlabel="Recall (cancelamentos encontrados)", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.01))
    a2.set_title("Precisão x Recall (conjunto de teste)")
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    fig.tight_layout(w_pad=3)
    return _salvar(fig, caminho)


def curva_de_lucro(limiares, lucros, limiar_escolhido, titulo, caminho=None):
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(limiares, lucros / 1000, color=AZUL)
    i = int(np.argmin(np.abs(limiares - limiar_escolhido)))
    ax.scatter([limiares[i]], [lucros[i] / 1000], s=70, color=AZUL, edgecolor=SUPERFICIE, linewidth=2, zorder=3)
    ax.annotate(
        f"limiar escolhido no treino: {br(limiares[i])}\nlucro no teste: US$ {br(lucros[i] / 1000, 1)} mil",
        (limiares[i], lucros[i] / 1000), xytext=(0.07, 22), textcoords="data", color=TEXTO, fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color=TEXTO_2, lw=0.8),
    )
    j = int(np.argmin(np.abs(limiares - 0.5)))
    ax.scatter([limiares[j]], [lucros[j] / 1000], s=60, color=SUPERFICIE, edgecolor=TEXTO_2, linewidth=1.5, zorder=3)
    ax.annotate(f"limiar padrão 0,50: US$ {br(lucros[j] / 1000, 1)} mil", (limiares[j], lucros[j] / 1000), xytext=(10, 10),
                textcoords="offset points", color=TEXTO_2, fontsize=9)
    ax.axhline(0, color=TEXTO_2, linewidth=0.8)
    ax.set_xlabel("Limiar de probabilidade para acionar a retenção")
    ax.set_ylabel("Lucro da campanha (US$ mil)")
    ax.set_title(titulo)
    return _salvar(fig, caminho)


def matriz_confusao(cm: np.ndarray, titulo: str, caminho=None):
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.grid(False)
    tons = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]
    ordem = np.argsort(np.argsort(cm.ravel())).reshape(cm.shape)
    rotulos = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]]
    for i in range(2):
        for j in range(2):
            cor = tons[ordem[i, j]]
            ax.add_patch(plt.Rectangle((j, i), 0.97, 0.97, color=cor))
            ink = "white" if ordem[i, j] >= 2 else TEXTO
            ax.text(j + 0.485, i + 0.42, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                    fontsize=17, fontweight="bold", color=ink)
            ax.text(j + 0.485, i + 0.65, rotulos[i][j], ha="center", va="center", fontsize=8.5, color=ink)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.485, 1.485], ["Previsto: fica", "Previsto: cancela"])
    ax.set_yticks([0.485, 1.485], ["Real: ficou", "Real: cancelou"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo)
    return _salvar(fig, caminho)


def importancia(importancias: pd.Series, titulo: str, caminho=None):
    """Importância por permutação (queda no ROC-AUC ao embaralhar cada coluna)."""
    aplicar_estilo()
    imp = importancias.rename(index=NOMES_COLUNAS).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=AZUL, height=0.62, edgecolor=SUPERFICIE, linewidth=2)
    for yi, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, yi, br(v, 3), va="center", fontsize=9, color=TEXTO)
    ax.set_xlim(0, imp.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Queda no ROC-AUC quando a coluna é embaralhada")
    ax.set_title(titulo)
    return _salvar(fig, caminho)
