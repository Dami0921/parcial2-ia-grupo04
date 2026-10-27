"""Segundo parcial IA - Grupo 04: movilidad en tramos viales de Cartago."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RUTA_CSV = BASE / "data" / "grupo_04.csv"
RUTA_LIMPIO = BASE / "data" / "grupo_04_limpio.csv"
COL_X1 = "velocidad_promedio_kmh"
COL_X2 = "flujo_vehiculos_hora"
COL_Y = "accidente_reportado"


def cargar_datos(ruta=RUTA_CSV):
    """Lee el CSV original sin modificarlo."""
    return pd.read_csv(ruta)


def inspeccionar(df):
    """Tarea 2: salidas de inspección pedidas en el enunciado."""
    print("=== Primeras 5 filas ===")
    print(df.head())
    print("\n=== Forma (filas, columnas) ===")
    print(df.shape)
    print("\n=== Tipos de datos ===")
    print(df.dtypes)
    print("\n=== Nulos por columna ===")
    print(df.isnull().sum())
    print("\n=== Filas duplicadas ===")
    print(df.duplicated().sum())
    print("\n=== Últimas 3 filas (busca filas de resumen) ===")
    print(df.tail(3))
    print("\n=== Estadísticas (busca valores imposibles) ===")
    print(df.describe())
    print("\n=== Categorías de zona ===")
    print(df["zona"].value_counts(dropna=False))


def limpiar_datos(df):
    """Tarea 3: detecta y corrige los problemas de calidad, mostrando evidencia."""
    filas_antes = len(df)
    df = df.copy()

    # Problema 1: fila de resumen TOTAL (no es un tramo real)
    resumen = df["tramo_id"].str.upper() == "TOTAL"
    print("\n[1] Filas de resumen detectadas:")
    print(df[resumen])
    df = df[~resumen].copy()
    print(f"    -> eliminadas: {resumen.sum()}")

    # Problema 2: categorías de zona escritas de formas distintas
    print("\n[2] Zonas antes:", sorted(df["zona"].unique()))
    df["zona"] = df["zona"].str.strip().str.title()
    print("    Zonas después:", sorted(df["zona"].unique()))

    # Problema 3: valor imposible/atípico en el flujo (regla del IQR)
    q1, q3 = df[COL_X2].quantile([0.25, 0.75])
    limite = q3 + 1.5 * (q3 - q1)
    atipicos = df[df[COL_X2] > limite]
    print(f"\n[3] Flujo máximo aceptable por IQR: {limite:.0f} veh/h")
    print(atipicos[["tramo_id", COL_X2]])
    media_antes = df[COL_X2].mean()
    std_antes = df[COL_X2].std()
    mediana_flujo = df.loc[df[COL_X2] <= limite, COL_X2].median()
    df.loc[df[COL_X2] > limite, COL_X2] = mediana_flujo
    print(f"    Media de {COL_X2}: antes {media_antes:.1f} | después {df[COL_X2].mean():.1f}")
    print(f"    Desv. estándar:    antes {std_antes:.1f} | después {df[COL_X2].std():.1f}")

    # Problema 4: valores faltantes en la velocidad
    nulos = df[df[COL_X1].isnull()]
    print(f"\n[4] Nulos en {COL_X1}:")
    print(nulos[["tramo_id", "zona", COL_X1]])
    mediana_vel = df[COL_X1].median()
    df[COL_X1] = df[COL_X1].fillna(mediana_vel)
    print(f"    -> rellenados con la mediana ({mediana_vel:.1f} km/h)")

    # Problema 5: tipo de dato del objetivo (float por el NaN de TOTAL)
    df[COL_Y] = df[COL_Y].astype(int)
    print(f"\n[5] {COL_Y} convertido a entero")

    print(f"\nFilas antes: {filas_antes} | después: {len(df)}")
    print("Registros por clase:")
    print(df[COL_Y].value_counts().sort_index())
    return df


def main():
    df = cargar_datos()
    inspeccionar(df)
    limpio = limpiar_datos(df)
    limpio.to_csv(RUTA_LIMPIO, index=False)
    print(f"\nGuardado: {RUTA_LIMPIO}")


if __name__ == "__main__":
    main()