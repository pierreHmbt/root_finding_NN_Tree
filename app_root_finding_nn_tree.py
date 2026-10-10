"""Application Streamlit — récupérer la racine de l'arbre du plus proche voisin.

On choisit le domaine (tore T^d ou cube [0,1]^d), la dimension d, le nombre de
points tirés n, puis l'application simule l'arbre T_n, calcule la centralité de
Jordan psi_n sur le graphe non étiqueté et affiche l'estimateur H_K de la
racine : les K sommets de plus petite valeur de psi_n.

Lancement :
    ~/.venv/bin/streamlit run app_racine_arbre_ppv.py
"""

import time

from itertools import product

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# L'algorithme
# ---------------------------------------------------------------------------


def simulate_nn_tree(points, torus=True, k0=8, workers=-1):
    """Arbre du plus proche voisin en ligne.

    parents[j] = argmin_{i<j} d(x_i, x_j) ; distance euclidienne périodique
    (``boxsize=1``) sur le tore, ordinaire sur le cube. Méthode exacte :
    recherche kNN par lots, avec doublement de k pour les points sans
    prédécesseur dans la liste.
    """
    n = len(points)
    parents = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return parents
    tree = cKDTree(points, boxsize=1.0 if torus else None)
    todo = np.arange(1, n)
    k = min(k0, n - 1)
    while todo.size:
        kq = min(k + 1, n)  # +1 : le point lui-même, à distance nulle
        _, idx = tree.query(points[todo], k=kq, workers=workers)
        idx = np.asarray(idx).reshape(todo.size, -1)
        mask = idx[:, 1:] < todo[:, None]
        has = mask.any(axis=1)
        first = mask.argmax(axis=1)
        sel = todo[has]
        parents[sel] = idx[has, first[has] + 1]
        todo = todo[~has]
        k *= 2
    return parents


def jordan_centrality(n, edges, return_subtree=False):
    """psi(v) = taille de la plus grande composante de T - v, en O(n).

    Enracine l'arbre en 0 (toute racine donne le même psi) et calcule les
    tailles de sous-arbres sub : psi(v) = max(max_c sub(c), n - sub(v)).
    """
    u = np.concatenate([edges[:, 0], edges[:, 1]])
    v = np.concatenate([edges[:, 1], edges[:, 0]])
    sorter = np.argsort(u, kind="stable")
    u, v = u[sorter], v[sorter]
    counts = np.bincount(u, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)])

    parent = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    chunks = [np.array([0], dtype=np.int64)]
    frontier = chunks[0]
    while frontier.size:
        starts = indptr[frontier]
        cf = indptr[frontier + 1] - starts
        total = int(cf.sum())
        if total == 0:
            break
        offsets = np.repeat(np.cumsum(cf) - cf, cf)
        idx = np.repeat(starts, cf) + np.arange(total) - offsets
        cand = v[idx]
        mask = ~visited[cand]
        newv = cand[mask]
        parent[newv] = np.repeat(frontier, cf)[mask]
        visited[newv] = True
        chunks.append(newv)
        frontier = newv
    order = np.concatenate(chunks)

    sub = np.ones(n, dtype=np.int64)
    for x in order[:0:-1]:
        sub[parent[x]] += sub[x]
    maxchild = np.zeros(n, dtype=np.int64)
    for x in order[1:]:
        p = parent[x]
        if sub[x] > maxchild[p]:
            maxchild[p] = sub[x]
    psi = np.maximum(maxchild, n - sub)
    return (psi, sub) if return_subtree else psi


def certified_k(psi, sub):
    """Certificat du lemme 2 de la note : plus petit M tel que, avec
    beta = 1 - psi(1)/n, on ait |D(v)| <= (3/4) beta n pour tout v > M.
    Le lemme garantit alors rang(racine) <= M, donc 1 dans H_M quel que soit
    le départage des ex æquo.
    """
    n = sub.size
    beta = 1.0 - psi[0] / n
    if beta <= 0.0:
        return n
    # sommets v >= 2 (1-indexés) violant (B) : |D(v)| > (3/4) beta n
    violateurs = np.flatnonzero(sub[1:] > 0.75 * beta * n) + 1   # labels 0-indexés
    return int(violateurs.max()) + 1 if violateurs.size else 1   # M = plus grand v violant


# ---------------------------------------------------------------------------
# Tracés
# ---------------------------------------------------------------------------


def _point_size(n):
    return float(np.clip(4000.0 / max(n, 1), 2.0, 35.0))


def _edge_segments(xy, parents, torus):
    """Segments d'arêtes, tracés périodiquement sur le tore.

    Chaque arête va du point à l'image du parent la plus proche ; si cette
    image sort du cube [0,1]^d, la copie translatée par ±1 (par coordonnée)
    est ajoutée : l'arête traverse le bord et réapparaît de l'autre côté.
    """
    child = xy[1:].astype(float)
    head = xy[parents[1:]].astype(float)
    if torus:
        delta = child - head
        delta = delta - np.round(delta)
        head = child - delta
    d = xy.shape[1]
    shifts = np.where(head < 0.0, 1.0, np.where(head > 1.0, -1.0, 0.0))
    segs = []
    for combo in product((-1.0, 0.0, 1.0), repeat=d):
        mask = np.all(shifts == np.array(combo), axis=1)
        if not mask.any():
            continue
        shift = np.array(combo)
        segs.append(np.stack([child[mask] + shift, head[mask] + shift], axis=1))
    return np.concatenate(segs, axis=0)


def _scatter_and_marks(ax, xy, psi, H, color_by_psi, s, is3d=False):
    """Nuage de points (coloré par psi si demandé) + racine et H_K en évidence."""
    if color_by_psi:
        c = psi
        if is3d:
            sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=c, cmap="viridis", s=s)
        else:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="viridis", s=s)
    else:
        if is3d:
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], color="steelblue", s=s, label="points")
        else:
            ax.scatter(xy[:, 0], xy[:, 1], color="steelblue", s=s, label="points")
        sc = None
    others = [v for v in H if v != 0]
    if others:
        args = (
            (xy[others, 0], xy[others, 1], xy[others, 2]) if is3d else (xy[others, 0], xy[others, 1])
        )
        ax.scatter(
            *args,
            s=2.0 * s + 24,
            facecolors="none",
            edgecolors="k",
            linewidths=1.0,
            zorder=4,
            label=r"$H_K\setminus\{1\}$",
        )
    root = (xy[0, 0], xy[0, 1], xy[0, 2]) if is3d else (xy[0, 0], xy[0, 1])
    if is3d:
        ax.scatter(
            [root[0]], [root[1]], [root[2]], s=200, marker="*", c="crimson",
            zorder=5, depthshade=False, label="racine 1",
        )
    else:
        ax.scatter(
            [root[0]], [root[1]], s=200, marker="*", c="crimson", zorder=5, label="racine 1"
        )
    return sc


def draw_domain(ax, points, parents, psi, H, torus, d, show_edges, color_by_psi):
    """Dessine les points, l'arbre et l'estimateur dans le domaine choisi."""
    n = len(points)
    s = _point_size(n)
    sc = None

    if d == 1 and torus:
        ang = 2.0 * np.pi * points[:, 0]
        xy = np.column_stack([np.cos(ang), np.sin(ang)])
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="0.85", lw=1.0, zorder=0)
        if show_edges and n > 1:
            segs = np.stack([xy[1:], xy[parents[1:]]], axis=1)
            ax.add_collection(LineCollection(segs, colors="0.6", lw=0.6, zorder=1))
        sc = _scatter_and_marks(ax, xy, psi, H, color_by_psi, s)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(r"Tore $\mathbb{T}^1$ (cercle)")

    elif d == 2:
        xy = points
        if show_edges and n > 1:
            segs = _edge_segments(xy, parents, torus)
            ax.add_collection(LineCollection(segs, colors="0.6", lw=0.6, zorder=1))
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.85", lw=1.0, zorder=0)
        m = 0.02
        sc = _scatter_and_marks(ax, xy, psi, H, color_by_psi, s)
        ax.set(xlim=(-m, 1 + m), ylim=(-m, 1 + m), aspect="equal")
        ax.set_title(
            r"Tore $\mathbb{T}^2$ (arêtes au plus court)" if torus else r"Cube $[0,1]^2$"
        )

    elif d == 3:
        xyz = points
        if show_edges and n > 1:
            segs = _edge_segments(xyz, parents, torus)
            ax.add_collection3d(
                Line3DCollection(list(segs), colors="0.6", lw=0.5, zorder=1)
            )
        m = 0.02
        sc = _scatter_and_marks(ax, xyz, psi, H, color_by_psi, s, is3d=True)
        ax.set(
            xlim=(-m, 1 + m),
            ylim=(-m, 1 + m),
            zlim=(-m, 1 + m),
            xlabel="x",
            ylabel="y",
            zlabel="z",
        )
        ax.set_title(
            r"Tore $\mathbb{T}^3$ (vue 3D)" if torus else r"Cube $[0,1]^3$ (vue 3D)"
        )

    else:  # d == 1, cube
        x = points[:, 0]
        base = 0.0
        ax.plot([0, 1], [base, base], color="0.85", lw=1.2, zorder=0)
        ytop = 0.15
        if show_edges and n > 1:
            t = np.linspace(0.0, 1.0, 12)
            x1, x2 = x[parents[1:]], x[1:]
            h = 0.25 * np.abs(x2 - x1) + 0.02
            ytop = float(max(0.15, 1.3 * h.max()))
            xs = x1[:, None] + t[None, :] * (x2 - x1)[:, None]
            ys = base + 4.0 * h[:, None] * t[None, :] * (1.0 - t[None, :])
            segs = np.stack([xs, ys], axis=-1)
            ax.add_collection(LineCollection(list(segs), colors="0.6", lw=0.6, zorder=1))
        xy = np.column_stack([x, np.full(n, base)])
        sc = _scatter_and_marks(ax, xy, psi, H, color_by_psi, s)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.03, ytop))
        ax.set_title(r"Cube $[0,1]$ (arêtes en arcs)")

    if sc is not None:
        ax.figure.colorbar(sc, ax=ax, shrink=0.75, label=r"$\psi_n$")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="best")
    return ax


def draw_psi(ax, psi, K, kcert, R):
    """Valeurs de psi triées : rang du sommet 1 et bornes de l'algorithme."""
    n = len(psi)
    order = np.argsort(psi, kind="stable")
    ax.plot(np.arange(1, n + 1), psi[order], lw=1.5, color="steelblue", label=r"$\psi_n$ trié")
    ax.axhline(psi[0], color="crimson", ls="--", lw=1.2, label=r"$\psi_n(1)$")
    beta = 1.0 - psi[0] / n
    ax.axhline(
        n - 0.75 * beta * n,
        color="darkorange",
        ls=":",
        lw=1.2,
        label=r"$n-\frac{3}{4}\beta n$",
    )
    ax.plot([R], [psi[0]], "o", color="crimson", ms=6, zorder=5)
    ax.axvline(K, color="grey", ls="-.", lw=1.2, label=f"K = {K}")
    ax.axvline(kcert, color="0.2", ls=":", lw=1.2, label=r"$K_{\rm cert}$ = " + f"{kcert}")
    ax.annotate(
        f"R = {R}",
        xy=(R, psi[0]),
        xytext=(0.10 * n, 0.5 * (psi[0] + float(psi.max()))),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        color="crimson",
    )
    ax.set(
        xlabel=r"sommets triés par $\psi_n$ croissant",
        ylabel=r"$\psi_n(v)$",
        title="Centralité de Jordan triée",
    )
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Données (simulation mémorisée)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def simulate_cached(domaine, d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, int(d)))
    parents = simulate_nn_tree(points, torus=(domaine == "tore"))
    return points, parents


@st.cache_data(show_spinner=False)
def analyse_cached(domaine, d, n, seed, n0):
    points, parents = simulate_cached(domaine, d, n, seed)
    pts = points[:n0].copy()
    par = parents[:n0].copy()
    edges = np.column_stack([np.arange(1, n0), par[1:]])
    t0 = time.perf_counter()
    psi, sub = jordan_centrality(n0, edges, return_subtree=True)
    dt = time.perf_counter() - t0
    return pts, par, psi, sub, dt


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Racine de l'arbre PPV — tore ou cube", layout="wide")
st.title("Récupérer la racine de l'arbre du plus proche voisin")
st.markdown(
    "Les points arrivent i.i.d. uniformes ; chacun se connecte au plus proche des points "
    "déjà présents (distance euclidienne, périodique sur le tore). La racine est le premier "
    "point, $1$. L'algorithme classe les sommets par **centralité de Jordan** "
    "$\\psi_n(v)$ (taille de la plus grande composante après suppression de $v$) et estime "
    "la racine par $H_K$ : les $K$ sommets de plus petite valeur."
)

if "seed" not in st.session_state:
    st.session_state.seed = 0


def _new_draw():
    st.session_state.seed = (int(st.session_state.seed) + 1) % 10**9


with st.sidebar:
    st.header("Paramètres")
    domaine_label = st.radio(
        "Domaine", [r"Tore $\mathbb{T}^d$", r"Cube $[0,1]^d$"], index=0, key="domaine"
    )
    domaine = "tore" if domaine_label.startswith("Tore") else "cube"
    d = st.selectbox("Dimension $d$", [1, 2, 3], index=1, key="dim")
    n = st.slider(
        "Nombre de points tirés $n$",
        min_value=50,
        max_value=30000,
        value=2000,
        step=50,
        key="n",
    )
    K = st.slider("Taille de l'estimateur $H_K$", 1, 100, 10, key="k")
    st.number_input("Graine", min_value=0, max_value=10**9, step=1, key="seed")
    st.button("🎲 Nouveau tirage", on_click=_new_draw)

    with st.expander("Affichage et préfixe"):
        show_edges = st.checkbox("Afficher les arêtes", value=True, key="edges")
        color_psi = st.checkbox(r"Colorer les points par $\psi_n$", value=False, key="color_psi")
        n0 = st.slider(
            "Points utilisés pour le calcul (préfixe)",
            min_value=20,
            max_value=n,
            value=n,
            step=10,
            key=f"n0_{n}",
        )

with st.spinner("Simulation de l'arbre et calcul de $\\psi_n$..."):
    pts, par, psi, sub, dt_psi = analyse_cached(domaine, d, n, st.session_state.seed, n0)

n0 = len(pts)
R = int(np.count_nonzero(psi <= psi[0]))
K_eff = int(min(K, n0))
H = np.argsort(psi, kind="stable")[:K_eff]
root_in_H = bool(0 in H)
kcert = certified_k(psi, sub)
v_min = int(np.argmin(psi))

domaine_txt = r"tore $\mathbb{T}^d$" if domaine == "tore" else r"cube $[0,1]^d$"
st.caption(
    f"Domaine : {domaine_txt} · d = {d} · points utilisés : {n0} (sur {n} tirés) · "
    f"calcul de $\\psi_n$ : {dt_psi * 1000:.0f} ms"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rang pessimiste $R$", R, help="nombre de sommets v tels que ψ(v) ≤ ψ(1)")
col2.metric("K choisi", K_eff)
col3.metric("$K_{\\rm cert}$ (lemme 2)", kcert, help="K prouvable pour cet arbre, plus conservateur")
col4.metric("$\\psi_n(1)/n$", f"{psi[0] / n0:.2f}")

if K_eff >= R:
    st.success(
        f"✅ La racine est **certifiée** dans $H_K$ : K = {K_eff} ≥ R = {R}, "
        "donc $1 \\in H_K$ quel que soit le départage des ex æquo."
    )
elif root_in_H:
    st.warning(
        f"⚠️ Le sommet 1 figure dans le $H_K$ affiché (départage favorable), mais "
        f"K = {K_eff} < R = {R} : aucune garantie."
    )
else:
    st.error(
        f"❌ La racine n'est pas dans $H_K$ : K = {K_eff} < R = {R}. "
        "La note garantit le succès pour K ≥ R sur cet arbre."
    )

fig = plt.figure(figsize=(13.5, 5.4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d") if d == 3 else fig.add_subplot(1, 2, 1)
draw_domain(ax1, pts, par, psi, H, domaine == "tore", d, show_edges, color_psi)
ax2 = fig.add_subplot(1, 2, 2)
draw_psi(ax2, psi, K_eff, kcert, R)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Estimateur fourni par l'algorithme")
if v_min == 0:
    st.markdown("Le sommet à plus petite centralité est **le sommet 1 : la racine elle-même**.")
else:
    st.markdown(
        f"Le sommet à plus petite centralité est {v_min + 1} — la racine (sommet 1) est "
        f"classée {R}ᵉ dans l'ordre pessimiste."
    )

rows = [
    {
        "sommet": int(v) + 1,
        "ψ": int(psi[v]),
        "ψ/n": round(float(psi[v]) / n0, 3),
        "est la racine": "★" if v == 0 else "",
    }
    for v in H
]
st.dataframe(pd.DataFrame(rows), hide_index=True)

with st.expander("Comment ça marche ? (et code de l'algorithme)"):
    st.markdown(
        r"""
**Le modèle.** Sur le tore $\mathbb{T}^d=\mathbb{R}^d/\mathbb{Z}^d$ (distance euclidienne
modulo $\mathbb{Z}^d$) ou sur le cube $[0,1]^d$, les points $x_1,x_2,\dots$ sont i.i.d.
uniformes. Pour $j\ge 2$, $p(j)=\arg\min_{i<j} d(x_i,x_j)$, et $T_n$ a pour arêtes
$\{j,p(j)\}$.

**La statistique.** $\psi_n(v)$ = taille de la plus grande composante connexe de
$T_n\setminus\{v\}$. Elle ne dépend que du graphe non étiqueté et se calcule en $O(n)$
(tailles de sous-arbres, enracinement arbitraire). L'estimateur est
$H_K$ = les $K$ plus petites valeurs.

**Le résultat** (note de P. Humbert et F. Laplante, 2026). Pour tout $d\ge 1$ et
$\varepsilon\in(0,1)$, il existe $K=K(d,\varepsilon)$ — indépendant de $n$ — tel que
$\liminf_n \mathbb{P}(1\in H_K)\ge 1-\varepsilon$ ; la même preuve vaut sur le cube.

**Le rang pessimiste.** $R=\#\{v:\psi_n(v)\le\psi_n(1)\}$. Si $K\ge R$, la racine est
dans $H_K$ pour *tout* départage des ex æquo ; c'est exactement ce que mesure la table
ci-dessus et ce que suit le curseur « préfixe » (les arbres sont emboîtés : le préfixe
est l'arbre des $n_0$ premiers points).

**Implémentation.** Simulation exacte de $T_n$ en $\tilde O(n)$ : plus proches voisins
par lots dans un arbre KD (`boxsize=1` pour le tore), avec raffinement de $k$. Calcul de
$\psi_n$ sur les seules arêtes, sans géométrie.
"""
    )
    st.code(
        """def jordan_centrality(n, edges, return_subtree=False):
    ...
    sub = np.ones(n, dtype=np.int64)
    for x in order[:0:-1]:          # ordre inverse du parcours en largeur
        sub[parent[x]] += sub[x]
    psi = np.maximum(maxchild, n - sub)
    return (psi, sub) if return_subtree else psi""",
        language="python",
    )
