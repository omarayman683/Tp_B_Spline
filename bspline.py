"""
TP - Reconstruction d'un contour par B-spline (fait main, sans librairie B-spline)

Courbe B-spline :   C(t) = somme_i  N(i, p, t) * P_i
  P_i : points de controle (ici = les points du contour)
  p   : degre
  N   : fonctions de base, calculees par la formule de Cox - de Boor
"""
import numpy as np
import matplotlib.pyplot as plt


def noeuds(n, p):
    """Vecteur de noeuds clampe : p+1 zeros, valeurs regulieres, p+1 uns.
    n = nombre de points de controle."""
    milieu = np.linspace(0, 1, n - p + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), milieu, np.ones(p + 1)])


def N(i, p, t, u):
    """Fonction de base B-spline (formule de Cox - de Boor)."""
    if p == 0:
        return 1.0 if u[i] <= t < u[i + 1] else 0.0
    a = 0.0
    if u[i + p] != u[i]:
        a = (t - u[i]) / (u[i + p] - u[i]) * N(i, p - 1, t, u)
    b = 0.0
    if u[i + p + 1] != u[i + 1]:
        b = (u[i + p + 1] - t) / (u[i + p + 1] - u[i + 1]) * N(i + 1, p - 1, t, u)
    return a + b


def bspline(P, p, nb=400):
    """Points de la courbe : pour chaque t, somme des P_i ponderes par N."""
    n = len(P)
    u = noeuds(n, p)
    courbe = []
    for t in np.linspace(0, 1, nb, endpoint=False):
        courbe.append(sum(N(i, p, t, u) * P[i] for i in range(n)))
    courbe.append(P[-1])  # en t = 1, la courbe clampee arrive sur le dernier point
    return np.array(courbe)


# 1) Lecture des points (colonnes : index, x, y)
P = np.loadtxt("data/points_contour.csv", delimiter=",", skiprows=1)[:, 1:]

# 2) Nuage de points
plt.plot(P[:, 0], P[:, 1], "o", color="orange")
plt.title("Nuage de points du contour")
plt.axis("equal")
plt.savefig("nuage_points.png", dpi=130)

# 3) Contour ferme : on recopie le premier point a la fin
P = np.vstack([P, P[0]])

# 4) B-spline pour les degres 1, 2, 3, 4
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, p in zip(axes.ravel(), [1, 2, 3, 4]):
    C = bspline(P, p)
    ax.plot(P[:, 0], P[:, 1], "o--", color="orange", label="points de controle (contour)")
    ax.plot(C[:, 0], C[:, 1], "r-", linewidth=2, label="courbe B-spline")
    ax.set_title(f"Degre p = {p}")
    ax.axis("equal")
    ax.legend(fontsize=8)
fig.savefig("resultat_bspline.png", dpi=130)
plt.show()
