import pandas as pd
import numpy as np

# =====================================================
# CONFIGURACIÓN
# =====================================================
archivo = "m92011_final.dat"

# FWHM mediano que usaste para m92011
fwhm = 3.355

# Criterio: rechazar si el vecino más cercano está dentro de 3*FWHM
limite = 3 * fwhm

# =====================================================
# LEER ARCHIVO
# El archivo debe tener columnas:
# XCENTER YCENTER MAG MERR
# =====================================================
df = pd.read_csv(
    archivo,
    delim_whitespace=True,
    names=["x", "y", "mag", "merr"],
    na_values=["INDEF"]
)

# Quitar filas sin coordenadas
df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

# =====================================================
# CALCULAR VECINO MÁS CERCANO
# =====================================================
coords = df[["x", "y"]].to_numpy()

# Matriz de distancias entre todas las estrellas
distancias = np.sqrt(
    ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
)

# Evitar que una estrella se compare consigo misma
np.fill_diagonal(distancias, np.inf)

# Distancia al vecino más cercano
df["dist_vecino_mas_cercano"] = distancias.min(axis=1)

# Índice del vecino más cercano
df["indice_vecino_mas_cercano"] = distancias.argmin(axis=1) + 1

# =====================================================
# APLICAR CRITERIO
# =====================================================
# Aceptadas: no tienen otra estrella dentro de 3*FWHM
aceptadas = df[df["dist_vecino_mas_cercano"] > limite].copy()

# Rechazadas: sí tienen otra estrella dentro de 3*FWHM
rechazadas = df[df["dist_vecino_mas_cercano"] <= limite].copy()

# =====================================================
# GUARDAR ARCHIVOS PARA TVMARK
# Solo X Y, sin encabezado
# =====================================================
aceptadas[["x", "y"]].to_csv(
    "m92011_aceptadas_aisladas.coo",
    sep=" ",
    index=False,
    header=False,
    float_format="%.3f"
)

rechazadas[["x", "y"]].to_csv(
    "m92011_rechazadas_vecino_cercano.coo",
    sep=" ",
    index=False,
    header=False,
    float_format="%.3f"
)

# Guardar tabla completa para el informe
df.to_csv("m92011_tabla_vecinos_cercanos.csv", index=False)

# =====================================================
# RESUMEN
# =====================================================
print("Archivo analizado:", archivo)
print("FWHM usado:", fwhm)
print("Límite 3*FWHM:", limite)
print("Total de estrellas:", len(df))
print("Aceptadas aisladas:", len(aceptadas))
print("Rechazadas por vecino cercano:", len(rechazadas))

print("\nArchivos creados:")
print("m92011_aceptadas_aisladas.coo")
print("m92011_rechazadas_vecino_cercano.coo")
print("m92011_tabla_vecinos_cercanos.csv")
