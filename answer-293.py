import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_cohorts_by_survival(df: pd.DataFrame, k: int) -> np.ndarray:
    cohorts = df['cohort'].unique()

    def kaplan_meier_median(group):
        times = np.sort(group['days_active'].unique())
        survival = 1.0
        median = None

        for t in times:
            at_risk = (group['days_active'] >= t).sum()
            events = ((group['days_active'] == t) & (group['churned'] == 1)).sum()

            if at_risk == 0:
                continue

            survival *= (1 - events / at_risk)

            if survival <= 0.5 and median is None:
                median = t

        return median if median is not None else times[-1]

    features = []

    for cohort in cohorts:
        group = df[df['cohort'] == cohort]
        median_surv = kaplan_meier_median(group)
        size = len(group)
        features.append([median_surv, size])

    features_scaled = StandardScaler().fit_transform(
        np.array(features, dtype=float)
    )

    labels = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    ).fit_predict(features_scaled)

    return labels
