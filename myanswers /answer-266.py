import pandas as pd
import numpy as np
import random


# FUNCIÓN PRINCIPAL


def resumen_ventas_por_categoria(df):
    """
    Retorna un resumen de ventas por categoría.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas:
        - producto
        - categoria
        - ventas

    Retorna
    -------
    pandas.DataFrame
        DataFrame con:
        - categoria
        - promedio_ventas
        - max_ventas
        - min_ventas
    """

    resultado = (
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

    return resultado



# GENERADOR DE CASO DE USO

def generar_caso_de_uso_resumen_ventas_por_categoria():
    """
    Genera un caso de uso aleatorio para probar
    la función resumen_ventas_por_categoria
    """

    # Cantidad aleatoria de registros
    n = random.randint(6, 15)

    # Categorías posibles
    categorias_posibles = ["A", "B", "C", "D"]

    # Productos posibles
    productos_posibles = [
        "prod_1",
        "prod_2",
        "prod_3",
        "prod_4",
        "prod_5",
        "prod_6",
        "prod_7",
        "prod_8",
        "prod_9",
        "prod_10"
    ]


    # GENERAR DATOS ALEATORIOS


    data = {
        "producto": np.random.choice(
            productos_posibles,
            size=n
        ),

        "categoria": np.random.choice(
            categorias_posibles,
            size=n
        ),

        "ventas": np.random.randint(
            10,
            500,
            size=n
        )
    }

    # Crear DataFrame
    df = pd.DataFrame(data)


    # INPUT

    input_data = {
        "df": df.copy()
    }


    # OUTPUT ESPERADO

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



# EJECUCIÓN DE PRUEBA

input_data, output_data = (
    generar_caso_de_uso_resumen_ventas_por_categoria()
)

# DataFrame original
df = input_data["df"]

print("\n================ DATAFRAME ORIGINAL ================\n")
print(df)

print("\n================ RESUMEN DE VENTAS ================\n")
print(output_data)
