import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def pipeline_svr(df, target_col, test_size):
    """
    Función que construye un pipeline de preprocesamiento
    y regresión usando SVR.
    """


    # 1. Separar características (X) y variable objetivo (y)
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()


    # 2. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )


    # 3. Construir el pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])


    # 4. Entrenar el pipeline
    pipeline.fit(X_train, y_train)


    # 5. Realizar predicciones
    y_pred = pipeline.predict(X_test)


    # 6. Calcular MAE
    mae = round(mean_absolute_error(y_test, y_pred), 4)


    # 7. Retornar resultados
    return y_pred, mae



# GENERADOR DEL CASO DE USO

def generar_caso_de_uso_pipeline_svr():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función pipeline_svr.
    """


    # 1. Configuración aleatoria
    n_filas = random.randint(50, 150)
    n_features = random.randint(2, 5)
    test_size = round(random.choice([0.2, 0.25, 0.3]), 2)


    # 2. Generar características aleatorias
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    X_data = np.random.uniform(
        0,
        10,
        size=(n_filas, n_features)
    )


    # 3. Generar variable objetivo
    coeficientes = np.random.uniform(
        1,
        5,
        size=n_features
    )

    ruido = np.random.normal(
        0,
        0.5,
        size=n_filas
    )

    y_data = (
        np.sin(X_data @ coeficientes / n_features) * 10
        + ruido
    )


    # 4. Construir DataFrame
    target_col = 'objetivo'

    df = pd.DataFrame(
        X_data,
        columns=feature_cols
    )

    df[target_col] = y_data


    # 5. INPUT
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'test_size': test_size
    }


    # 6. OUTPUT ESPERADO
    y_pred, mae = pipeline_svr(
        df,
        target_col,
        test_size
    )

    output_data = (y_pred, mae)

    return input_data, output_data


# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":

    # Generar caso de prueba
    i, o = generar_caso_de_uso_pipeline_svr()

    # Mostrar INPUTS
    print("----------- INPUTS -----------")

    for k, v in i.items():
        print(f"\n{k}:\n")
        print(v)

    # Mostrar OUTPUTS
    print("\n----------- OUTPUT ESPERADO -----------")

    y_pred, mae = o

    print("\nPredicciones (y_pred):\n")
    print(y_pred)

    print("\nMAE:\n")
    print(mae)
