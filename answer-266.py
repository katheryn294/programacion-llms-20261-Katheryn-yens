import pandas as pd


def resumen_ventas_por_categoria(df):
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
