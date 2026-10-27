# parcial2-ia-grupo04 – Segundo Parcial de Inteligencia Artificial

**Grupo:** 04
**Integrantes:** Damián Correa Tusarma, Saray Foronda Restrepo
**Archivo de datos:** `grupo_04.csv` (Movilidad – tramos viales de Cartago)
**Entradas:** `velocidad_promedio_kmh`, `flujo_vehiculos_hora` | **Objetivo:** `accidente_reportado` (1 = hubo accidente, 0 = no)

## Tarea 1: Configuración del entorno
Stack: Python 3.12, NumPy, Pandas, Matplotlib y Seaborn (sin librerías de Machine Learning). Contenedor: `parcial2_ia`.

Reproducir el entorno (WSL + Docker):
```bash
git clone https://github.com/Dami0921/parcial2-ia-grupo04.git
cd parcial2-ia-grupo04
docker stop python_ia            # si el contenedor de las clases está encendido
docker compose up -d --build
docker exec -it parcial2_ia bash

# dentro del contenedor
cd src
python parcial2.py
```
Capturas: `capturas/` (contenedor en ejecución y versiones).

## Tarea 2: Carga e inspección
El script `src/parcial2.py` muestra las primeras 5 filas, la forma (81, 5), los tipos, los nulos (zona 1, velocidad 3, accidente 1) y los duplicados (0).

**Pregunta clave: ¿alguna columna numérica aparece con un tipo que no debería tener?**
Sí: `accidente_reportado` aparece como `float64` cuando es una etiqueta binaria que debería ser entera. Ocurre porque la fila `TOTAL` trae un valor vacío (NaN) en esa columna, y Pandas convierte a decimal las columnas con valores faltantes.

## Tarea 3: Calidad y limpieza de datos
| # | Problema | Cómo se detectó | Decisión | Por qué |
|---|---|---|---|---|
| 1 | Fila de resumen `TOTAL` | `tail()` muestra `TOTAL` con zona y objetivo vacíos; `describe()` daba velocidad máx. 4428 | Eliminar | No es un tramo vial; sus sumas distorsionan medias y escalado |
| 2 | Zonas escritas distinto (`NORTE`, `SUR`, `vía panamericana`, `" Sur"`, `"Norte "`, `oriente`) | `value_counts()` mostró 11 etiquetas para 5 zonas | `strip()` y `title()`