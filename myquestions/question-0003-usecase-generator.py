import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer

def generar_caso_de_uso_preparar_matriz_con_imputacion():
    """
    Genera un caso de uso aleatorio para la función
    preparar_matriz_con_imputacion
    """

    n_filas = random.randint(6, 15)
    n_features = random.randint(2, 5)

    data = np.random.randn(n_filas, n_features)
    columnas = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=columnas)

    # Introducir NaN aleatorios
    mask = np.random.choice([True, False], size=df.shape, p=[0.2, 0.8])
    df = df.mask(mask)

    target_col = "target"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    output_data = (X_imputed, y)

    return input_data, output_data
