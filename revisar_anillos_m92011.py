import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# CONFIGURACIÓN
# ============================================================

imagen_fits = "m92011.fits"

# Puedes usar:
# "m92011_final.dat" para todas las estrellas
# o "m92011_aceptadas_aisladas.coo" para revisar solo las que pasaron vecinos cercanos
archivo_coords = "m92011_final.dat"

# Parámetros usados en la fotometría
fwhm = 3.355
apertura = 3 * fwhm       # 10.065 pix

# Anillo de cielo usado en IRAF / phot / qphot
annulus = 15              # radio interno del anillo
dannulus = 10             # grosor del anillo
r_in = annulus
r_out = annulus + dannulus

# Criterio de rechazo por contaminación en el anillo
# Si más del 3% de los píxeles del anillo son outliers altos, se rechaza
frac_outliers_limite = 0.03

# ============================================================
# LEER IMAGEN FITS
# ============================================================

data = fits.getdata(imagen_fits)

# Si la imagen tiene más dimensiones, tomar la primera capa
if data.ndim > 2:
    data = data[0]

data = np.asarray(data, dtype=float)

# ============================================================
# LEER COORDENADAS
# ============================================================

# Si el archivo es .dat con columnas x y mag merr
# o .coo con columnas x y, esto funciona para ambos casos
coords = pd.read_csv(
    archivo_coords,
    sep=r"\s+",
    header=None,
    comment="#",
    engine="python"
)

coords = coords.iloc[:, :2]
coords.columns = ["x", "y"]
coords = coords.dropna().reset_index(drop=True)

# Agregar ID desde 1 para compararlo con IRAF/tvmark
coords["id"] = np.arange(1, len(coords) + 1)

# ============================================================
# FUNCIÓN PARA EXTRAER PÍXELES DEL ANILLO
# ============================================================

def extraer_anillo(data, x_iraf, y_iraf, r_in, r_out):
    """
    x_iraf, y_iraf vienen de IRAF y están en coordenadas 1-indexadas.
    Python usa índices 0-indexados, por eso se resta 1.
    """

    x0 = x_iraf - 1
    y0 = y_iraf - 1

    ny, nx = data.shape

    xmin = int(max(np.floor(x0 - r_out), 0))
    xmax = int(min(np.ceil(x0 + r_out) + 1, nx))
    ymin = int(max(np.floor(y0 - r_out), 0))
    ymax = int(min(np.ceil(y0 + r_out) + 1, ny))

    sub = data[ymin:ymax, xmin:xmax]

    yy, xx = np.indices(sub.shape)
    xx = xx + xmin
    yy = yy + ymin

    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)

    mask_anillo = (rr >= r_in) & (rr <= r_out)

    valores = sub[mask_anillo]
    valores = valores[np.isfinite(valores)]

    return valores

# ============================================================
# ANALIZAR ANILLOS
# ============================================================

estadisticas = []
valores_para_boxplot = []
ids_para_boxplot = []

for _, row in coords.iterrows():
    x = row["x"]
    y = row["y"]
    star_id = int(row["id"])

    valores = extraer_anillo(data, x, y, r_in, r_out)

    if len(valores) < 10:
        continue

    q1 = np.percentile(valores, 25)
    mediana = np.percentile(valores, 50)
    q3 = np.percentile(valores, 75)
    iqr = q3 - q1

    limite_superior = q3 + 1.5 * iqr

    outliers_altos = valores[valores > limite_superior]
    n_outliers_altos = len(outliers_altos)
    frac_outliers_altos = n_outliers_altos / len(valores)

    p95 = np.percentile(valores, 95)
    maximo = np.max(valores)

    contaminada = frac_outliers_altos > frac_outliers_limite

    estadisticas.append({
        "id": star_id,
        "x": x,
        "y": y,
        "n_pix_anillo": len(valores),
        "q1": q1,
        "mediana_cielo": mediana,
        "q3": q3,
        "iqr": iqr,
        "limite_superior_boxplot": limite_superior,
        "n_outliers_altos": n_outliers_altos,
        "frac_outliers_altos": frac_outliers_altos,
        "p95": p95,
        "maximo": maximo,
        "contaminada": contaminada
    })

    valores_para_boxplot.append(valores)
    ids_para_boxplot.append(star_id)

stats = pd.DataFrame(estadisticas)

# ============================================================
# SEPARAR ACEPTADAS Y RECHAZADAS
# ============================================================

rechazadas = stats[stats["contaminada"]].copy()
aceptadas = stats[~stats["contaminada"]].copy()

# Guardar archivos .coo para marcar en IRAF
rechazadas[["x", "y"]].to_csv(
    "m92011_rechazadas_anillo_cielo.coo",
    sep=" ",
    index=False,
    header=False,
    float_format="%.3f"
)

aceptadas[["x", "y"]].to_csv(
    "m92011_aceptadas_anillo_cielo.coo",
    sep=" ",
    index=False,
    header=False,
    float_format="%.3f"
)

# Guardar tabla completa
stats.to_csv("m92011_estadisticas_anillos_cielo.csv", index=False)

# ============================================================
# DIAGRAMA DE CAJA
# ============================================================

# Para que sea legible, ordenamos por fracción de outliers altos
stats_ordenado = stats.sort_values("frac_outliers_altos", ascending=False)

valores_ordenados = []
labels_ordenados = []

for star_id in stats_ordenado["id"]:
    idx = ids_para_boxplot.index(star_id)
    valores_ordenados.append(valores_para_boxplot[idx])
    labels_ordenados.append(str(star_id))

plt.figure(figsize=(16, 7))
plt.boxplot(
    valores_ordenados,
    showfliers=True,
    flierprops=dict(marker='.', markersize=2)
)

plt.xticks(
    ticks=np.arange(1, len(labels_ordenados) + 1),
    labels=labels_ordenados,
    rotation=90,
    fontsize=6
)

plt.xlabel("ID de estrella, ordenado por posible contaminación")
plt.ylabel("Cuentas en el anillo de cielo")
plt.title("Diagrama de caja de los anillos de cielo en m92011")
plt.tight_layout()
plt.savefig("m92011_boxplot_anillos_cielo.png", dpi=200)
plt.close()

# ============================================================
# GRÁFICA MÁS CLARA: FRACCIÓN DE OUTLIERS ALTOS
# ============================================================

plt.figure(figsize=(12, 6))
plt.scatter(stats["id"], stats["frac_outliers_altos"], s=25)
plt.axhline(frac_outliers_limite, linestyle="--")

plt.xlabel("ID de estrella")
plt.ylabel("Fracción de píxeles outliers altos en el anillo")
plt.title("Criterio de contaminación del anillo de cielo")
plt.tight_layout()
plt.savefig("m92011_outliers_anillos_cielo.png", dpi=200)
plt.close()

# ============================================================
# RESUMEN
# ============================================================

print("Imagen:", imagen_fits)
print("Archivo de coordenadas:", archivo_coords)
print("FWHM:", fwhm)
print("Apertura 3*FWHM:", apertura)
print("Anillo de cielo interno:", r_in)
print("Anillo de cielo externo:", r_out)
print("Criterio de rechazo: frac_outliers_altos >", frac_outliers_limite)
print()
print("Total de estrellas analizadas:", len(stats))
print("Aceptadas por anillo de cielo:", len(aceptadas))
print("Rechazadas por anillo contaminado:", len(rechazadas))
print()
print("Archivos creados:")
print("m92011_boxplot_anillos_cielo.png")
print("m92011_outliers_anillos_cielo.png")
print("m92011_estadisticas_anillos_cielo.csv")
print("m92011_rechazadas_anillo_cielo.coo")
print("m92011_aceptadas_anillo_cielo.coo")
