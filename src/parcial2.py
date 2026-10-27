"""Segundo parcial IA - Grupo 04: movilidad en tramos viales de Cartago."""
from pathlib import Path

import pandas as pd

RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "grupo_04.csv"


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


def main():
    df = cargar_datos()
    inspeccionar(df)


if __name__ == "__main__":
    main()