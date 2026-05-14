import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import VarianceThreshold


# FUNCIÓN PRINCIPAL

def detectar_columnas_baja_cardinalidad(df, umbral):
    """
    Detecta columnas con baja cardinalidad y
    columnas numéricas con varianza cero.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada

    umbral : int
        Número mínimo de valores únicos permitido

    Retorna
    -------
    list
        Lista de nombres de columnas
    """


    # 1. DETECTAR BAJA CARDINALIDAD

    cols_baja_card = [
        col
        for col in df.columns
        if df[col].nunique() < umbral
    ]


    # 2. SELECCIONAR COLUMNAS NUMÉRICAS


    df_num = df.select_dtypes(include=[np.number])

    # Lista para columnas con varianza cero
    cols_var_cero = []


    # 3. APLICAR VARIANCETHRESHOLD

    if not df_num.empty:

        selector = VarianceThreshold(
            threshold=0
        )

        selector.fit(df_num)

        # True -> columna válida
        # False -> varianza cero
        soporte = selector.get_support()

        cols_var_cero = [
            col
            for col, keep in zip(df_num.columns, soporte)
            if not keep
        ]


    # 4. UNIR RESULTADOS

    resultado = list(
        set(cols_baja_card + cols_var_cero)
    )

    return resultado


# GENERADOR DE CASOS DE USO

def generar_caso_de_uso_detectar_columnas_baja_cardinalidad():
    """
    Genera un caso de prueba aleatorio
    para la función
    detectar_columnas_baja_cardinalidad
    """


    # PARÁMETROS ALEATORIOS

    n_rows = random.randint(6, 12)

    n_cols = random.randint(3, 5)


    # GENERAR DATAFRAME

    data = {}

    for i in range(n_cols):

        col_name = f"col_{i}"

        tipo = random.choice([
            "baja",
            "media",
            "alta"
        ])


        # BAJA CARDINALIDAD

        if tipo == "baja":

            valores = np.random.choice(
                range(3),
                size=n_rows
            )


        # MEDIA CARDINALIDAD

        elif tipo == "media":

            valores = np.random.choice(
                range(10),
                size=n_rows
            )


        # ALTA CARDINALIDAD
        else:

            valores = np.random.randint(
                0,
                100,
                size=n_rows
            )

        data[col_name] = valores


    # CREAR DATAFRAME
    df = pd.DataFrame(data)


    # DEFINIR UMBRAL
    umbral = random.randint(2, 4)


    # ASEGURAR UNA COLUMNA DE BAJA CARDINALIDAD
    if all(
        df[col].nunique() >= umbral
        for col in df.columns
    ):

        col_random = random.choice(df.columns)

        df[col_random] = np.random.choice(
            [1, 1, 2],
            size=n_rows
        )


    # INPUT
    input_data = {
        "df": df.copy(),
        "umbral": umbral
    }


    # OUTPUT ESPERADO

    # A. Columnas con baja cardinalidad
    cols_baja_card = [
        col
        for col in df.columns
        if df[col].nunique() < umbral
    ]

    # B. Columnas numéricas
    df_num = df.select_dtypes(include=[np.number])

    # C. Columnas con varianza cero
    cols_var_cero = []

    if not df_num.empty:

        varianzas = df_num.var()

        cols_var_cero = (
            varianzas[varianzas == 0]
            .index
            .tolist()
        )

    # D. Unión final
    output_data = list(
        set(cols_baja_card + cols_var_cero)
    )

    return input_data, output_data



# EJECUCIÓN DE PRUEBA

if __name__ == "__main__":

    entrada, salida = (
        generar_caso_de_uso_detectar_columnas_baja_cardinalidad()
    )


    # MOSTRAR INPUT

    print("=============== INPUT ===============\n")

    print(entrada["df"])

    print("\nUmbral:", entrada["umbral"])


    # MOSTRAR OUTPUT ESPERADO

    print("\n=============== OUTPUT ESPERADO ===============\n")

    print(salida)


    # PROBAR FUNCIÓN

    resultado = detectar_columnas_baja_cardinalidad(
        entrada["df"],
        entrada["umbral"]
    )

    print("\n=============== RESULTADO FUNCIÓN ===============\n")

    print(resultado)
