"""
TP - Reconstruction d'un contour par B-spline
=============================================

Objectif : a partir d'un ensemble de points (le contour d'une piece),
construire "a la main" une courbe B-spline, SANS utiliser de fonction
magique type scipy.interpolate.BSpline.

Tout repose sur UNE seule formule (la recursion de Cox - de Boor) qui
definit les fonctions de base B-spline. On programme cette formule, puis
la courbe est simplement une somme ponderee des points de controle.

                 n
      C(t)  =   somme   N_{i,p}(t) * P_i
                i=0

ou :
  - P_i          : les points de controle (ici = les points du contour)
  - p            : le degre de la courbe (1, 2, 3, 4...)
  - N_{i,p}(t)   : les "fonctions de base", calculees par Cox - de Boor
  - t            : le parametre qui parcourt la courbe

Seules numpy (tableaux) et matplotlib (affichage) sont utilisees.
Aucune fonction de B-spline toute faite.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) LECTURE DES POINTS DU CONTOUR
# ---------------------------------------------------------------------------
def lire_points(chemin):
    """
    Lit un fichier (CSV ou TXT) de la forme :  index , x , y
    et renvoie deux listes numpy : les x et les y.

    Le separateur peut etre ',' (csv) ou ';' (txt) : on gere les deux.
    La premiere ligne est l'entete (index,x,y) : on la saute.
    """
    xs, ys = [], []
    with open(chemin, "r") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne == "":
                continue
            # on remplace ';' par ',' pour traiter les deux formats pareil
            ligne = ligne.replace(";", ",")
            colonnes = ligne.split(",")
            # on saute l'entete (ligne qui ne commence pas par un nombre)
            try:
                _index = int(colonnes[0])
            except ValueError:
                continue
            xs.append(float(colonnes[1]))
            ys.append(float(colonnes[2]))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# 2) LE VECTEUR DE NOEUDS (knot vector)
# ---------------------------------------------------------------------------
def vecteur_noeuds(n, p):
    """
    Construit un vecteur de noeuds "clampe" (ouvert uniforme).

    n  : indice du dernier point de controle (il y a donc n+1 points)
    p  : degre de la courbe

    Regle (la plus simple et la plus classique) :
      - p+1 noeuds valent 0 au debut   -> la courbe PART du 1er point
      - p+1 noeuds valent 1 a la fin    -> la courbe ARRIVE au dernier point
      - les noeuds du milieu sont repartis regulierement entre 0 et 1

    Longueur totale du vecteur de noeuds : n + p + 2
    """
    m = n + p + 1               # indice du dernier noeud
    noeuds = np.zeros(m + 1)    # il y a m+1 noeuds au total

    for i in range(m + 1):
        if i < p + 1:
            noeuds[i] = 0.0                      # debut "colle" a 0
        elif i > n:
            noeuds[i] = 1.0                      # fin "collee" a 1
        else:
            noeuds[i] = (i - p) / (n - p + 1)    # milieu regulier
    return noeuds


# ---------------------------------------------------------------------------
# 3) LES FONCTIONS DE BASE : recursion de Cox - de Boor  (LE COEUR DU TP)
# ---------------------------------------------------------------------------
def base_bspline(i, p, t, noeuds):
    """
    Calcule la valeur de la fonction de base N_{i,p}(t).

    C'est la DEFINITION meme d'une B-spline, programmee directement.

    --- Cas de base (degre 0) ---
      N_{i,0}(t) = 1 si  noeuds[i] <= t < noeuds[i+1]
                 = 0 sinon
      (c'est une "marche" : vaut 1 sur un seul intervalle)

    --- Cas recursif (degre p) ---
      N_{i,p}(t) =  (t - u_i)/(u_{i+p} - u_i)       * N_{i,p-1}(t)
                 + (u_{i+p+1} - t)/(u_{i+p+1} - u_{i+1}) * N_{i+1,p-1}(t)

    Un terme dont le denominateur est 0 est simplement ignore (= 0),
    c'est la convention habituelle qui evite les divisions par zero.
    """
    # --- degre 0 : la marche ---
    if p == 0:
        # Remarque : on inclut la borne de droite au tout dernier intervalle
        # pour que t = 1 (le dernier point) soit bien pris en compte.
        if (noeuds[i] <= t < noeuds[i + 1]) or \
           (t == noeuds[-1] and noeuds[i] <= t <= noeuds[i + 1]):
            return 1.0
        return 0.0

    # --- degre p : on combine deux fonctions de degre p-1 ---
    gauche = 0.0
    denom_gauche = noeuds[i + p] - noeuds[i]
    if denom_gauche > 0:
        gauche = (t - noeuds[i]) / denom_gauche * base_bspline(i, p - 1, t, noeuds)

    droite = 0.0
    denom_droite = noeuds[i + p + 1] - noeuds[i + 1]
    if denom_droite > 0:
        droite = (noeuds[i + p + 1] - t) / denom_droite * base_bspline(i + 1, p - 1, t, noeuds)

    return gauche + droite


# ---------------------------------------------------------------------------
# 4) EVALUATION DE LA COURBE B-SPLINE EN UN POINT
# ---------------------------------------------------------------------------
def point_courbe(t, px, py, p, noeuds):
    """
    Calcule le point C(t) = somme_i  N_{i,p}(t) * P_i

    px, py : coordonnees des points de controle
    Renvoie (x, y) : le point de la courbe pour le parametre t.
    """
    n = len(px) - 1
    x = 0.0
    y = 0.0
    for i in range(n + 1):
        b = base_bspline(i, p, t, noeuds)   # poids du point de controle i
        x += b * px[i]
        y += b * py[i]
    return x, y


def courbe_bspline(px, py, p, nb_points=300):
    """
    Construit toute la courbe B-spline en echantillonnant le parametre t
    entre 0 et 1 (nb_points valeurs).

    Renvoie deux tableaux (cx, cy) : la courbe reconstruite.
    """
    n = len(px) - 1
    noeuds = vecteur_noeuds(n, p)

    ts = np.linspace(0.0, 1.0, nb_points)   # t de 0 a 1
    cx, cy = [], []
    for t in ts:
        x, y = point_courbe(t, px, py, p, noeuds)
        cx.append(x)
        cy.append(y)
    return np.array(cx), np.array(cy)


# ---------------------------------------------------------------------------
# 5) PROGRAMME PRINCIPAL : lire, construire, afficher, comparer
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- lecture des points du contour ---
    px, py = lire_points("data/points_contour.csv")
    print("Nombre de points du contour :", len(px))

    # --- TP section 4 : afficher d'abord les points sous forme de nuage ---
    plt.figure(figsize=(7, 6))
    plt.plot(px, py, "o", color="orange", markersize=5)
    plt.title("Nuage de points du contour (donnees brutes)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nuage_points.png", dpi=130)
    print("Figure enregistree : nuage_points.png")

    # Pour un contour FERME : on recopie le premier point a la fin
    # (sinon la courbe clampee s'arrete au dernier point et ne "boucle" pas).
    px = np.append(px, px[0])
    py = np.append(py, py[0])

    # --- on teste plusieurs degres, comme demande dans le TP (p=1,2,3,4) ---
    degres = [1, 2, 3, 4]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    for ax, p in zip(axes, degres):
        cx, cy = courbe_bspline(px, py, p, nb_points=400)

        # points du contour / de controle (ici ce sont les memes points)
        ax.plot(px, py, "o--", color="lightgray", markerfacecolor="orange",
                markersize=5, linewidth=1, label="points de controle (= contour)")

        # la courbe B-spline reconstruite
        ax.plot(cx, cy, "-", color="crimson", linewidth=2.2,
                label="courbe B-spline reconstruite")

        ax.set_title("Degre p = %d" % p)
        ax.axis("equal")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reconstruction d'un contour par B-spline (Cox - de Boor, fait main)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("resultat_bspline.png", dpi=130)
    print("Figure enregistree : resultat_bspline.png")
    # plt.show()   # decommenter pour afficher dans une fenetre
