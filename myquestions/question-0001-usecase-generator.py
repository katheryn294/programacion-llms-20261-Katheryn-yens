import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_seleccionar_productos():
    """
    Genera un caso de uso aleatorio para la función seleccionar_productos
    """

    n = random.randint(5, 15)

    data = {
        "price": np.random.randint(50, 200, size=n),
        "category": np.random.choice(["A", "B", "C"], size=n)
    }

    incluir_margin = random.choice([True, False])

    if incluir_margin:
        data["margin"] = np.random.randint(5, 20, size=n)

    df = pd.DataFrame(data)

    df.index = np.arange(1000, 1000 + n)

    input_data = {
        "df": df.copy()
    }

    resultado = []

    for idx, row in df.iterrows():
        cumple = False

        if row["price"] > 100:
            cumple = True

        if "margin" in df.columns:
            if row["margin"] > 12:
                cumple = True

        if cumple:
            resultado.append(idx)

    output_data = resultado

    return input_data, output_data
