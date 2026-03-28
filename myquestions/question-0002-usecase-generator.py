import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_resumen_ventas_por_categoria():
    """
    Genera un caso de uso aleatorio para la función
    resumen_ventas_por_categoria
    """

    n = random.randint(6, 15)

    categorias_posibles = ["A", "B", "C", "D"]
    productos_posibles = [
        "prod_1", "prod_2", "prod_3", "prod_4", "prod_5",
        "prod_6", "prod_7", "prod_8", "prod_9", "prod_10"
    ]

    data = {
        "producto": np.random.choice(productos_posibles, size=n),
        "categoria": np.random.choice(categorias_posibles, size=n),
        "ventas": np.random.randint(10, 500, size=n)
    }

    df = pd.DataFrame(data)

    input_data = {
        "df": df.copy()
    }

    output_data = (
        df.groupby("categoria")["ventas"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={
            "mean": "promedio_ventas",
            "max": "max_ventas",
            "min": "min_ventas"
        })
        .sort_values("categoria")
        .reset_index(drop=True)
    )

    return input_data, output_data
