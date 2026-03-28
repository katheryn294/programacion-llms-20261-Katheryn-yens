import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_y_evaluar_knn():
    """
    Genera un caso de uso aleatorio para la función
    entrenar_y_evaluar_knn
    """

    n_muestras = random.randint(20, 40)
    n_features = random.randint(2, 5)

    X = np.random.randn(n_muestras, n_features)
    y = np.random.randint(0, 2, size=n_muestras)

    train_pct = random.choice([0.6, 0.7, 0.75, 0.8])
    n_neighbors = random.choice([1, 3, 5])

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "train_pct": train_pct,
        "n_neighbors": n_neighbors
    }

    Xtr, Xts, ytr, yts = train_test_split(
        X, y, train_size=train_pct, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(Xtr, ytr)

    preds = model.predict(Xts)
    acc = accuracy_score(yts, preds)

    output_data = {
        "Xts": Xts,
        "yts": yts,
        "preds": preds,
        "accuracy": acc
    }

    return input_data, output_data
