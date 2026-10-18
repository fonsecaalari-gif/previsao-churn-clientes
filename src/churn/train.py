"""Treino, comparação de modelos, escolha do limiar e avaliação final.

Uso:
    python -m churn.train

Etapas:
1. Divide os dados em treino (80%) e teste (20%), estratificado. O teste só é
   tocado no final.
2. Compara 4 abordagens com validação cruzada estratificada (5 folds) no treino.
3. Escolhe o melhor modelo pela Precisão Média (PR-AUC), métrica adequada para
   classes desbalanceadas (~27% de churn).
4. Escolhe o limiar de decisão que maximiza o lucro usando previsões
   *fora da amostra* (out-of-fold) do treino, sem usar o teste.
5. Treina o modelo final no treino inteiro e avalia uma única vez no teste.
6. Salva modelo, métricas (JSON) e gráficos.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import plots
from churn.business import curva_lucro, lucro_campanha, melhor_limiar
from churn.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from churn.data import load_dataset
from churn.features import build_preprocessor, feature_engineering_step


def construir_modelos() -> dict[str, Pipeline]:
    """Cada modelo é um Pipeline completo: dados brutos -> probabilidade."""

    def pipe(estimador, escalar=True):
        return Pipeline(
            [
                ("features", feature_engineering_step()),
                ("prep", build_preprocessor(scale_numeric=escalar)),
                ("modelo", estimador),
            ]
        )

    return {
        "Baseline (sem modelo)": pipe(DummyClassifier(strategy="prior")),
        "Regressão Logística": pipe(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)
        ),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            escalar=False,
        ),
        "XGBoost": pipe(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            escalar=False,
        ),
    }


def main() -> dict:
    inicio = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_tr):,} clientes | Teste: {len(X_te):,} clientes | churn {y.mean():.1%}")

    # ------------------------------------------------------------------ 2. CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    modelos = construir_modelos()
    tabela_cv = []
    for nome, modelo in modelos.items():
        r = cross_validate(modelo, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
        linha = {
            "modelo": nome,
            "roc_auc_media": r["test_roc_auc"].mean(),
            "roc_auc_desvio": r["test_roc_auc"].std(),
            "pr_auc_media": r["test_average_precision"].mean(),
            "pr_auc_desvio": r["test_average_precision"].std(),
        }
        tabela_cv.append(linha)
        print(f"  {nome:<24} ROC-AUC {linha['roc_auc_media']:.3f} ± {linha['roc_auc_desvio']:.3f} | "
              f"PR-AUC {linha['pr_auc_media']:.3f} ± {linha['pr_auc_desvio']:.3f}")
    tabela_cv = pd.DataFrame(tabela_cv).sort_values("pr_auc_media", ascending=False)

    # --------------------------------------------------------- 3. melhor modelo
    candidatos = tabela_cv[~tabela_cv["modelo"].str.startswith("Baseline")]
    nome_melhor = candidatos.iloc[0]["modelo"]
    print(f"\nMelhor modelo na validação cruzada: {nome_melhor}")

    # ------------------------------------------------- 4. limiar pelo lucro (OOF)
    proba_oof = cross_val_predict(
        clone(modelos[nome_melhor]), X_tr, y_tr, cv=cv, method="predict_proba"
    )[:, 1]
    limiar, lucro_oof = melhor_limiar(y_tr.values, proba_oof, X_tr["MonthlyCharges"].values)
    print(f"Limiar que maximiza o lucro (out-of-fold, treino): {limiar:.2f}")

    # ---------------------------------------------- 5. treino final + teste
    probas_teste = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_tr, y_tr)
        probas_teste[nome] = modelo.predict_proba(X_te)[:, 1]
    final = modelos[nome_melhor]
    proba = probas_teste[nome_melhor]
    pred = (proba >= limiar).astype(int)
    pred_05 = (proba >= 0.5).astype(int)
    mensal = X_te["MonthlyCharges"].values

    lucro_modelo = lucro_campanha(y_te.values, pred, mensal)
    lucro_05 = lucro_campanha(y_te.values, pred_05, mensal)
    lucro_todos = lucro_campanha(y_te.values, np.ones_like(pred), mensal)
    lucro_aleatorio = lucro_campanha(  # mesmo nº de abordagens, escolhidas ao acaso
        y_te.values,
        np.random.default_rng(RANDOM_STATE).permutation(pred),
        mensal,
    )
    cm = confusion_matrix(y_te, pred)

    metricas = {
        "modelo_escolhido": nome_melhor,
        "limiar_decisao": limiar,
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "taxa_churn": float(y.mean()),
        "validacao_cruzada": tabela_cv.round(4).to_dict(orient="records"),
        "teste": {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "recall": float(recall_score(y_te, pred)),
            "precisao": float(precision_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "clientes_abordados": int(pred.sum()),
            "pct_base_abordada": float(pred.mean()),
            "matriz_confusao": cm.tolist(),
            "roc_auc_por_modelo": {n: float(roc_auc_score(y_te, p)) for n, p in probas_teste.items()},
            "pr_auc_por_modelo": {n: float(average_precision_score(y_te, p)) for n, p in probas_teste.items()},
        },
        "negocio_teste": {
            "lucro_modelo_limiar_otimo": lucro_modelo,
            "lucro_modelo_limiar_0_5": lucro_05,
            "lucro_abordar_todos": lucro_todos,
            "lucro_abordagem_aleatoria_mesmo_volume": lucro_aleatorio,
            "lucro_nao_fazer_nada": 0.0,
        },
    }

    # --------------------------------------------------------- 6. artefatos
    joblib.dump({"pipeline": final, "limiar": limiar, "modelo": nome_melhor}, MODELS_DIR / "modelo_churn.joblib")

    plots.curvas_roc_pr(probas_teste, y_te, FIGURES_DIR / "05_curvas_roc_pr.png")
    plots.matriz_confusao(cm, f"Matriz de confusão, limiar {plots.br(limiar)}", FIGURES_DIR / "06_matriz_confusao.png")
    lim, luc = curva_lucro(y_te.values, proba, mensal)
    plots.curva_de_lucro(lim, luc, limiar, "Quanto a campanha de retenção lucra em cada limiar",
                         FIGURES_DIR / "07_curva_lucro.png")

    imp = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                 random_state=RANDOM_STATE, n_jobs=1)
    importancias = pd.Series(imp.importances_mean, index=X_te.columns)
    metricas["importancia_permutacao"] = importancias.sort_values(ascending=False).round(4).to_dict()
    plots.importancia(importancias, "O que mais pesa na previsão de churn", FIGURES_DIR / "08_importancia.png")

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    t = metricas["teste"]
    print(f"\nTESTE ({nome_melhor}, limiar {limiar:.2f}): ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | "
          f"recall {t['recall']:.1%} | precisão {t['precisao']:.1%}")
    print(f"Lucro: modelo US$ {lucro_modelo:,.0f} | limiar 0,5 US$ {lucro_05:,.0f} | "
          f"todos US$ {lucro_todos:,.0f} | aleatório US$ {lucro_aleatorio:,.0f}")
    print(f"Concluído em {time.time() - inicio:.0f}s")
    return metricas


if __name__ == "__main__":
    main()
