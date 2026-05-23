import pandas as pd
import numpy as np


def detectar_columnas_baja_cardinalidad(df, umbral):
    cols_baja_card = [
        col for col in df.columns
        if df[col].nunique() < umbral
    ]

    df_num = df.select_dtypes(include=[np.number])

    cols_var_cero = []
    if not df_num.empty:
        varianzas = df_num.var()
        cols_var_cero = varianzas[varianzas == 0].index.tolist()

    resultado = list(set(cols_baja_card + cols_var_cero))

    return resultado
