"""Segundo parcial IA - Grupo 04: movilidad en tramos viales de Cartago."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # guarda gráficos sin pantalla (contenedor)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
RUTA_CSV = BASE / "data" / "grupo_04.csv"
RUTA_LIMPIO = BASE / "data" / "grupo_04_limpio.csv"
RUTA_GRAFICO1 = BASE / "data" / "grafico1_dispersion.png"
RUTA_GRAFICO2 = BASE / "data" / "grafico2_boxplot_velocidad.png"
RUTA_CURVA = BASE / "data" / "curva_aprendizaje.png"
RUTA_FRONTERA = BASE / "data" / "frontera_decision.png"
COL_X1 = "velocidad_promedio_kmh"
COL_X2 = "flujo_vehiculos_hora"
COL_Y = "accidente_reportado"
EPOCAS = 1000
TASA = 0.1


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


def graficar(df):
    """Tarea 4: dos gráficos con Seaborn guardados en data/."""
    sns.set_theme(style="whitegrid")

    # Gráfico 1: dispersión de las dos entradas coloreada por el objetivo
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=COL_X1, y=COL_X2, hue=COL_Y, palette="Set1", s=70)
    plt.title("Velocidad vs flujo de vehículos según accidente reportado")
    plt.xlabel("Velocidad promedio (km/h)")
    plt.ylabel("Flujo de vehículos (veh/h)")
    plt.legend(title="Accidente (1 = sí)", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO1, dpi=150)
    plt.close()

    # Gráfico 2: velocidad promedio en tramos con y sin accidente
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x=COL_Y, y=COL_X1, hue=COL_Y, palette="Set1", legend=False)
    plt.title("Velocidad promedio por clase de accidente")
    plt.xlabel("Accidente reportado (0 = no, 1 = sí)")
    plt.ylabel("Velocidad promedio (km/h)")
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO2, dpi=150)
    plt.close()
    print(f"Gráficos guardados: {RUTA_GRAFICO1.name}, {RUTA_GRAFICO2.name}")


def estandarizar(X):
    """Escala cada columna a media 0 y desviación 1; devuelve también media y desviación."""
    media = X.mean(axis=0)
    desviacion = X.std(axis=0)
    return (X - media) / desviacion, media, desviacion


def sigmoide(z):
    """Convierte cualquier número en una probabilidad entre 0 y 1."""
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(X, w, b):
    """Probabilidad de accidente: sigmoide(X @ w + b)."""
    return sigmoide(X @ w + b)


def perdida_logaritmica(y, p):
    """Entropía cruzada binaria promedio (con recorte para evitar log(0))."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def exactitud(y, p):
    """Proporción de aciertos usando umbral 0.5."""
    return np.mean((p >= 0.5).astype(int) == y)


def entrenar(X, y, tasa=TASA, epocas=EPOCAS, semilla=42):
    """Descenso del gradiente. Devuelve w, b e historiales de pérdida y exactitud."""
    rng = np.random.default_rng(semilla)
    w = rng.normal(0, 0.1, X.shape[1])
    b = 0.0
    w0, b0 = w.copy(), b
    perdidas, exactitudes = [], []
    n = len(y)
    for _ in range(epocas):
        p = predecir_probabilidad(X, w, b)
        perdidas.append(perdida_logaritmica(y, p))
        exactitudes.append(exactitud(y, p))
        error = p - y
        w = w - tasa * (X.T @ error) / n
        b = b - tasa * error.mean()
    return w, b, w0, b0, perdidas, exactitudes


def graficar_curva(perdidas):
    """curva_aprendizaje.png: pérdida por época."""
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(perdidas) + 1), perdidas, color="tab:blue")
    plt.title("Curva de aprendizaje del perceptrón")
    plt.xlabel("Época")
    plt.ylabel("Pérdida logarítmica")
    plt.tight_layout()
    plt.savefig(RUTA_CURVA, dpi=150)
    plt.close()


def graficar_frontera(df, w, b, media, desviacion):
    """frontera_decision.png: puntos de ambas clases y la frontera en unidades originales."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=COL_X1, y=COL_X2, hue=COL_Y, palette="Set1", s=70)
    x1 = np.linspace(df[COL_X1].min(), df[COL_X1].max(), 100)
    # Frontera: w1*(x1-m1)/s1 + w2*(x2-m2)/s2 + b = 0, despejando x2
    x2 = media[1] - desviacion[1] * (b + w[0] * (x1 - media[0]) / desviacion[0]) / w[1]
    plt.plot(x1, x2, color="black", linestyle="--", label="Frontera (p = 0.5)")
    plt.ylim(df[COL_X2].min() - 100, df[COL_X2].max() + 100)
    plt.title("Frontera de decisión del perceptrón")
    plt.xlabel("Velocidad promedio (km/h)")
    plt.ylabel("Flujo de vehículos (veh/h)")
    plt.legend(title="Accidente (1 = sí)", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(RUTA_FRONTERA, dpi=150)
    plt.close()


def perceptron(df):
    """Tarea 5: perceptrón con sigmoide y descenso del gradiente usando solo NumPy."""
    X = df[[COL_X1, COL_X2]].to_numpy(dtype=float)
    y = df[COL_Y].to_numpy(dtype=int)
    X_esc, media, desviacion = estandarizar(X)
    print("\n=== Estandarización ===")
    print(f"Media (velocidad, flujo):      {media.round(2)}")
    print(f"Desviación (velocidad, flujo): {desviacion.round(2)}")

    w, b, w0, b0, perdidas, exactitudes = entrenar(X_esc, y)
    print(f"\n=== Entrenamiento ({EPOCAS} épocas, tasa {TASA}) ===")
    print(f"Pesos iniciales: w1={w0[0]:.4f}, w2={w0[1]:.4f}, b={b0:.4f}")
    print(f"Pesos finales:   w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    print(f"Pérdida inicial: {perdidas[0]:.4f} | final: {perdidas[-1]:.4f}")
    print(f"Exactitud inicial: {exactitudes[0]:.4f} | final: {exactitudes[-1]:.4f}")

    graficar_curva(perdidas)
    graficar_frontera(df, w, b, media, desviacion)
    print(f"Gráficos guardados: {RUTA_CURVA.name}, {RUTA_FRONTERA.name}")
    return w, b, media, desviacion



def interpretar(df, w, b, media, desviacion):
    """Tarea 6: errores de clasificación y predicción de dos registros nuevos."""
    X = df[[COL_X1, COL_X2]].to_numpy(dtype=float)
    y = df[COL_Y].to_numpy(dtype=int)
    p = predecir_probabilidad((X - media) / desviacion, w, b)
    pred = (p >= 0.5).astype(int)
    mal = pred != y
    print(f"\n=== Registros del lado equivocado: {mal.sum()} de {len(y)} ===")
    print(f"Falsos positivos (predijo 1, real 0): {((pred == 1) & (y == 0)).sum()}")
    print(f"Falsos negativos (predijo 0, real 1): {((pred == 0) & (y == 1)).sum()}")
    errores = df[mal].copy()
    errores["probabilidad"] = p[mal].round(2)
    print(errores.to_string())

    print("\n=== Dos registros nuevos (escalados con media y desviación del entrenamiento) ===")
    nuevos = {
        "Esperado 1 (rápido y con mucho flujo)": [85, 1200],
        "Esperado 0 (lento y con poco flujo)": [35, 300],
    }
    for nombre, valores in nuevos.items():
        z = (np.array(valores, dtype=float) - media) / desviacion
        prob = predecir_probabilidad(z, w, b)
        print(f"{nombre}: velocidad={valores[0]} km/h, flujo={valores[1]} veh/h "
              f"-> z={z.round(2)}, probabilidad de accidente={prob:.4f}")



def main():
    df = cargar_datos()
    inspeccionar(df)
    limpio = limpiar_datos(df)
    limpio.to_csv(RUTA_LIMPIO, index=False)
    print(f"\nGuardado: {RUTA_LIMPIO}")
    graficar(limpio)
    w, b, media, desviacion = perceptron(limpio)
    interpretar(limpio, w, b, media, desviacion)

if __name__ == "__main__":
    main()