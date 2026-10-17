import pandas as pd

from churn.data import clean, load_raw, split_xy


def test_dataset_tem_formato_esperado():
    df = load_raw()
    assert df.shape == (7043, 21)
    assert df["customerID"].is_unique


def test_total_charges_em_branco_sao_clientes_novos():
    """Os 11 TotalCharges vazios devem ser todos de clientes com tenure 0."""
    df = load_raw()
    em_branco = df["TotalCharges"].astype(str).str.strip() == ""
    assert em_branco.sum() == 11
    assert (df.loc[em_branco, "tenure"] == 0).all()


def test_clean_gera_tipos_corretos_e_sem_nulos():
    df = clean(load_raw())
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert set(df["Churn"].unique()) == {0, 1}
    assert set(df["SeniorCitizen"].unique()) == {"Yes", "No"}
    assert df.isna().sum().sum() == 0


def test_clean_nao_altera_o_original():
    bruto = load_raw()
    copia = bruto.copy()
    clean(bruto)
    pd.testing.assert_frame_equal(bruto, copia)


def test_split_remove_id_e_alvo():
    X, y = split_xy(clean(load_raw()))
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == len(y)
