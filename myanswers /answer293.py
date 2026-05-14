import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random



# FUNCIÓN PRINCIPAL

def cluster_cohorts_by_survival(df: pd.DataFrame, k: int) -> np.ndarray:
    """
    Agrupa cohortes según:
    - Mediana de supervivencia Kaplan-Meier
    - Tamaño de la cohorte

    Parámetros
    ----------
    df : pd.DataFrame
        Columnas:
        - cohort
        - days_active
        - churned

    k : int
        Número de clusters

    Retorna
    -------
    np.ndarray
        Etiquetas de cluster por cohorte
    """

    # Cohortes únicas en el orden original
    cohorts = df['cohort'].unique()


    # ESTIMADOR KAPLAN-MEIER MANUAL

    def kaplan_meier_median(group):

        # Tiempos únicos ordenados
        times = np.sort(group['days_active'].unique())

        # Supervivencia acumulada inicial
        survival = 1.0

        # Mediana de supervivencia
        median = None

        # Recorrer tiempos
        for t in times:

            # Individuos en riesgo
            at_risk = (group['days_active'] >= t).sum()

            # Eventos observados en t
            events = (
                (group['days_active'] == t) &
                (group['churned'] == 1)
            ).sum()

            # Validación
            if at_risk == 0:
                continue

            # Kaplan-Meier
            survival *= (1 - events / at_risk)

            # Buscar momento donde S(t) <= 0.5
            if survival <= 0.5 and median is None:
                median = t

        # Si nunca baja de 0.5
        if median is None:
            median = times[-1]

        return median


    # CREAR FEATURES

    features = []

    for cohort in cohorts:

        # Filtrar cohorte
        group = df[df['cohort'] == cohort]

        # Mediana de supervivencia
        median_survival = kaplan_meier_median(group)

        # Tamaño de cohorte
        cohort_size = len(group)

        # Guardar features
        features.append([
            median_survival,
            cohort_size
        ])

    # Convertir a numpy array
    features = np.array(features, dtype=float)


    # ESCALADO

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(features)


    # KMEANS

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(features_scaled)

    return labels



# GENERADOR DEL CASO DE USO


def generar_caso_de_uso_cluster_cohorts_by_survival():

    random.seed(random.randint(0, 99999))
    np.random.seed(random.randint(0, 99999))

    # Número de cohortes
    n_cohorts = random.randint(3, 7)

    # Número de clusters
    k = random.randint(2, min(n_cohorts, 4))


    # GENERAR NOMBRES DE COHORTES


    seen = set()
    cohort_names = []

    while len(cohort_names) < n_cohorts:

        candidate = f"C{random.randint(2020, 2025)}Q{random.randint(1, 4)}"

        if candidate not in seen:
            seen.add(candidate)
            cohort_names.append(candidate)


    # GENERAR DATOS

    rows = []

    for cohort in cohort_names:

        # Número de jugadores
        n_players = random.randint(20, 80)

        # Perfil de supervivencia
        scale = random.uniform(10, 120)

        for _ in range(n_players):

            # Distribución exponencial
            days = int(np.random.exponential(scale)) + 1

            # Limitar máximo
            days = min(days, 365)

            # Evento de churn
            churned = 1 if random.random() < 0.75 else 0

            rows.append({
                'cohort': cohort,
                'days_active': days,
                'churned': churned
            })

    # Crear DataFrame
    df = pd.DataFrame(rows)


    # EJECUTAR FUNCIÓN PRINCIPAL


    labels = cluster_cohorts_by_survival(df, k)

    return {'df': df, 'k': k}, labels


# EJECUCIÓN DE PRUEBA


caso, labels = generar_caso_de_uso_cluster_cohorts_by_survival()

df = caso['df']
k = caso['k']

print("\n================ DATAFRAME ================\n")
print(df.head())

print("\n================ K ================\n")
print(k)

print("\n================ COHORTES ================\n")
print(df['cohort'].unique())

print("\n================ LABELS ================\n")
print(labels)
 
