import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# ARCHIVOS DE ENTRADA
# =========================================================
archivo_V = "m92011_final.dat"   # filtro V
archivo_B = "m92015_final.dat"   # filtro B

archivo_rech_anillo = "m92011_rechazadas_anillo_cielo.coo"
archivo_rech_vecino = "m92011_rechazadas_vecino_cercano.coo"

# tolerancia para comparar coordenadas
tol_match_cmd = 2.0      # para emparejar B y V
tol_categoria = 1.0      # para decidir si una estrella está en una lista .coo

# =========================================================
# FUNCIONES
# =========================================================
def leer_iraf_dat(nombre):
    filas = []
    with open(nombre, "r", encoding="utf-8", errors="replace") as f:
        for linea in f:
            partes = linea.split()
            if len(partes) >= 4:
                x = float(partes[0])
                y = float(partes[1])
                mag = np.nan if partes[2].upper() == "INDEF" else float(partes[2])
                merr = np.nan if partes[3].upper() == "INDEF" else float(partes[3])
                filas.append((x, y, mag, merr))
    df = pd.DataFrame(filas, columns=["x", "y", "mag", "merr"])
    return df

def leer_coo(nombre):
    df = pd.read_csv(
        nombre,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python"
    )
    df = df.iloc[:, :2]
    df.columns = ["x", "y"]
    return df.dropna().reset_index(drop=True)

def estimar_offset(V, B):
    dx_all, dy_all = [], []

    for _, vr in V.iterrows():
        dx = B["x"].values - vr["x"]
        dy = B["y"].values - vr["y"]

        mask = (dx > -15) & (dx < 15) & (dy > -15) & (dy < 15)
        dx_all.extend(dx[mask])
        dy_all.extend(dy[mask])

    dx_all = np.array(dx_all)
    dy_all = np.array(dy_all)

    binsx = np.arange(-15, 15.5, 0.5)
    binsy = np.arange(-15, 15.5, 0.5)

    H, xe, ye = np.histogram2d(dx_all, dy_all, bins=[binsx, binsy])
    imax = np.unravel_index(np.argmax(H), H.shape)

    dx0 = 0.5 * (xe[imax[0]] + xe[imax[0] + 1])
    dy0 = 0.5 * (ye[imax[1]] + ye[imax[1] + 1])

    return dx0, dy0

def refinar_offset(V, B, dx0, dy0, tol=3):
    matches = []

    for _, vr in V.iterrows():
        dist = np.sqrt(
            (B["x"].values - (vr["x"] + dx0))**2 +
            (B["y"].values - (vr["y"] + dy0))**2
        )
        j = np.argmin(dist)

        if dist[j] < tol:
            br = B.iloc[j]
            matches.append((br["x"] - vr["x"], br["y"] - vr["y"]))

    matches = np.array(matches)
    dx_ref = np.median(matches[:, 0])
    dy_ref = np.median(matches[:, 1])

    return dx_ref, dy_ref

def emparejar_BV(V, B, dx, dy, tol=2.0):
    candidatos = []

    for i, vr in V.iterrows():
        dist = np.sqrt(
            (B["x"].values - (vr["x"] + dx))**2 +
            (B["y"].values - (vr["y"] + dy))**2
        )
        for j, d in enumerate(dist):
            if d < tol:
                candidatos.append((d, i, j))

    candidatos = sorted(candidatos)

    usados_i = set()
    usados_j = set()
    matches = []

    for d, i, j in candidatos:
        if i not in usados_i and j not in usados_j:
            usados_i.add(i)
            usados_j.add(j)

            vr = V.iloc[i]
            br = B.iloc[j]

            matches.append({
                "x_V": vr["x"],
                "y_V": vr["y"],
                "V": vr["mag"],
                "err_V": vr["merr"],
                "x_B": br["x"],
                "y_B": br["y"],
                "B": br["mag"],
                "err_B": br["merr"],
                "match_distance_pix": d,
                "B_minus_V": br["mag"] - vr["mag"]
            })

    return pd.DataFrame(matches)

def pertenece_a_lista(x, y, lista_coords, tol=1.0):
    if len(lista_coords) == 0:
        return False
    dist = np.sqrt((lista_coords["x"].values - x)**2 + (lista_coords["y"].values - y)**2)
    return np.min(dist) <= tol

# =========================================================
# LEER DATOS
# =========================================================
V = leer_iraf_dat(archivo_V).dropna().reset_index(drop=True)
B = leer_iraf_dat(archivo_B).dropna().reset_index(drop=True)

rech_anillo = leer_coo(archivo_rech_anillo)
rech_vecino = leer_coo(archivo_rech_vecino)

# =========================================================
# EMPAREJAR ESTRELLAS ENTRE B Y V
# =========================================================
dx0, dy0 = estimar_offset(V, B)
dx, dy = refinar_offset(V, B, dx0, dy0, tol=3)

cmd = emparejar_BV(V, B, dx, dy, tol=tol_match_cmd)

# =========================================================
# ASIGNAR CATEGORÍAS
# Regla:
# 1) si está en rechazadas por anillo -> "Rechazada por anillo"
# 2) si no, pero está en rechazadas por vecino -> "Rechazada por vecino"
# 3) si no está en ninguna -> "Restante"
# =========================================================
categorias = []

for _, row in cmd.iterrows():
    x = row["x_V"]
    y = row["y_V"]

    es_anillo = pertenece_a_lista(x, y, rech_anillo, tol=tol_categoria)
    es_vecino = pertenece_a_lista(x, y, rech_vecino, tol=tol_categoria)

    if es_anillo:
        categorias.append("Rechazada por anillo")
    elif es_vecino:
        categorias.append("Rechazada por vecino")
    else:
        categorias.append("Restante")

cmd["categoria"] = categorias

# =========================================================
# FILTRO OPCIONAL DE ERRORES MUY GRANDES
# =========================================================
cmd_plot = cmd[
    (cmd["err_V"] < 0.30) &
    (cmd["err_B"] < 0.30) &
    (cmd["B_minus_V"] > -0.6) &
    (cmd["B_minus_V"] < 2.0)
].copy()

# =========================================================
# GUARDAR TABLA
# =========================================================
cmd.to_csv("M92_CMD_categorias_completo.csv", index=False)
cmd_plot.to_csv("M92_CMD_categorias_plot.csv", index=False)

# =========================================================
# GRAFICAR CMD
# =========================================================
plt.figure(figsize=(7, 8))

grupos = [
    ("Rechazada por anillo", "red"),
    ("Rechazada por vecino", "orange"),
    ("Restante", "blue")
]

for nombre, color in grupos:
    sub = cmd_plot[cmd_plot["categoria"] == nombre]
    plt.scatter(
        sub["B_minus_V"],
        sub["V"],
        s=20,
        alpha=0.8,
        label=f"{nombre} ({len(sub)})",
        color=color
    )

plt.gca().invert_yaxis()
plt.xlabel("Color instrumental B - V")
plt.ylabel("Magnitud instrumental V")
plt.title("CMD de M92 clasificado por filtros de calidad")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("M92_CMD_coloreado_filtros.png", dpi=200)
plt.show()

# =========================================================
# RESUMEN
# =========================================================
print("Offset estimado entre B y V:")
print(f"dx = {dx:.3f} pix, dy = {dy:.3f} pix")
print()

print("Conteos en CMD completo:")
print(cmd["categoria"].value_counts())
print()

print("Conteos en CMD filtrado para la gráfica:")
print(cmd_plot["categoria"].value_counts())
print()

print("Archivos creados:")
print("M92_CMD_coloreado_filtros.png")
print("M92_CMD_categorias_completo.csv")
print("M92_CMD_categorias_plot.csv")
