import os
import tempfile

cache_matplotlib = os.path.join(tempfile.gettempdir(), "lab5_matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_matplotlib, "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_matplotlib, "xdg"))

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
    sep=r"\s+",
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

# Guardar pares únicos de estrellas separadas por menos de 3*FWHM.
pares_cercanos = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        distancia = np.sqrt(
            (df.loc[i, "x"] - df.loc[j, "x"])**2 +
            (df.loc[i, "y"] - df.loc[j, "y"])**2
        )

        if distancia <= limite:
            pares_cercanos.append({
                "id_estrella_1": i + 1,
                "x_1": df.loc[i, "x"],
                "y_1": df.loc[i, "y"],
                "id_estrella_2": j + 1,
                "x_2": df.loc[j, "x"],
                "y_2": df.loc[j, "y"],
                "distancia_pix": distancia
            })

pares_cercanos = pd.DataFrame(pares_cercanos)
pares_cercanos.to_csv("m92011_pares_vecino_cercano.csv", index=False)

# =====================================================
# GRAFICAR NÚMERO DE ESTRELLAS VS RADIO DE APERTURA
# Conteo por intervalos de radio, no acumulativo.
# =====================================================
delta_radio = 1.0
dist_vecino = df["dist_vecino_mas_cercano"].to_numpy()
radio_max = np.ceil(dist_vecino.max() / delta_radio) * delta_radio
bordes_radio = np.arange(0, radio_max + delta_radio, delta_radio)
numero_estrellas, _ = np.histogram(dist_vecino, bins=bordes_radio)
radios_inicio = bordes_radio[:-1]
radios_fin = bordes_radio[1:]
radios_centro = radios_inicio + delta_radio / 2
estrellas_en_limite = (df["dist_vecino_mas_cercano"] <= limite).sum()

tabla_radio_vecino = pd.DataFrame({
    "radio_inicio_pix": radios_inicio,
    "radio_fin_pix": radios_fin,
    "radio_centro_pix": radios_centro,
    "delta_radio_pix": delta_radio,
    "n_estrellas_en_delta": numero_estrellas
})
tabla_radio_vecino.to_csv(
    "m92011_numero_estrellas_vs_radio_vecino_cercano.csv",
    index=False
)

plt.figure(figsize=(8, 5))
ax = plt.gca()

ax.bar(
    radios_inicio,
    numero_estrellas,
    width=delta_radio,
    align="edge",
    color="tab:blue",
    edgecolor="black",
    linewidth=0.4,
    alpha=0.85,
    label=f"Delta = {delta_radio:.1f} pix"
)
ax.axvline(
    limite,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label=f"Radio usado = {limite:.3f} pix"
)
ax.set_xlabel("Radio de apertura [pixeles]")
ax.set_ylabel("Número de estrellas en cada delta")
ax.set_title("Estrellas por intervalo de distancia al vecino más cercano")
ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
plt.savefig("m92011_numero_estrellas_vs_radio_vecino_cercano.png", dpi=200)
plt.close()

# =====================================================
# GRAFICAR VECINO MÁS CERCANO DE CADA ESTRELLA
# =====================================================
indices_vecinos = df["indice_vecino_mas_cercano"].to_numpy(dtype=int) - 1
coords_vecinos = coords[indices_vecinos]
segmentos = np.stack([coords, coords_vecinos], axis=1)

colores_segmentos = np.where(
    df["dist_vecino_mas_cercano"].to_numpy() <= limite,
    "tab:red",
    "0.65"
)

plt.figure(figsize=(8, 8))
ax = plt.gca()

lineas = LineCollection(
    segmentos,
    colors=colores_segmentos,
    linewidths=0.8,
    alpha=0.55
)
ax.add_collection(lineas)

ax.scatter(
    aceptadas["x"],
    aceptadas["y"],
    s=5,
    color="tab:blue",
    alpha=0.8,
    label=f"Aceptadas ({len(aceptadas)})"
)

ax.scatter(
    rechazadas["x"],
    rechazadas["y"],
    s=5,
    color="tab:red",
    alpha=0.9,
    label=f"Rechazadas por vecino cercano ({len(rechazadas)})"
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X [pixeles]")
ax.set_ylabel("Y [pixeles]")
ax.set_title("Vecino mas cercano de cada estrella en m92011")
ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
plt.savefig("m92011_vecinos_mas_cercanos.png", dpi=200)
plt.close()

# Detalle de las rechazadas, con IDs para ver los pares o grupos cercanos.
plt.figure(figsize=(8, 8))
ax = plt.gca()

segmentos_rechazadas = segmentos[rechazadas.index.to_numpy()]
lineas_rechazadas = LineCollection(
    segmentos_rechazadas,
    colors="tab:red",
    linewidths=1.5,
    alpha=0.7
)
ax.add_collection(lineas_rechazadas)

ax.scatter(
    rechazadas["x"],
    rechazadas["y"],
    s=42,
    color="tab:red",
    alpha=0.85
)

for idx, row in rechazadas.iterrows():
    star_id = idx + 1
    ax.annotate(
        str(star_id),
        (row["x"], row["y"]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color="black"
    )

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X [pixeles]")
ax.set_ylabel("Y [pixeles]")
ax.set_title("Detalle de estrellas rechazadas por vecino cercano")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("m92011_rechazadas_vecinos_detalle.png", dpi=200)
plt.close()

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
print("m92011_pares_vecino_cercano.csv")
print("m92011_numero_estrellas_vs_radio_vecino_cercano.csv")
print("m92011_numero_estrellas_vs_radio_vecino_cercano.png")
print("m92011_vecinos_mas_cercanos.png")
print("m92011_rechazadas_vecinos_detalle.png")
