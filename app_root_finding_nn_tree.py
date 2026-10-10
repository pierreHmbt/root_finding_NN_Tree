"""Streamlit app: recovering the root of the nearest neighbor tree.

Choose the domain (torus T^d or cube [0,1]^d), the dimension d and the number
of sampled points n; the app then simulates the tree T_n, computes the Jordan
centrality psi_n on the unlabeled graph and displays the root estimator H_K:
the K vertices with the smallest values of psi_n.

Run with:
    streamlit run app_root_finding_nn_tree.py
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numba import njit
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


def simulate_nn_tree(points: np.ndarray, torus: bool) -> np.ndarray:
    """Simulates the online nearest neighbor tree.

    Exact method: batched kNN queries, doubling ``k`` for the points whose
    neighbors all arrived later.

    Args:
        points (np.ndarray): Points in ``[0, 1]^d`` with shape ``(n, d)``.
        torus (bool): Whether the distance is periodic.

    Returns:
        np.ndarray: ``parents[j] = argmin_{i<j} d(x_i, x_j)``, with shape ``(n,)``
            and ``parents[0] = -1``.
    """
    n = len(points)
    parents = np.full(n, -1, dtype=np.int64)
    tree = cKDTree(points, boxsize=1.0 if torus else None)
    todo = np.arange(1, n)
    k = 8
    while todo.size:
        _, idx = tree.query(points[todo], k=min(k, n), workers=-1)
        older = idx < todo[:, None]  # also discards the point itself
        found = older.any(axis=1)
        parents[todo[found]] = idx[found, older[found].argmax(axis=1)]
        todo = todo[~found]
        k *= 2
    return parents


@njit(cache=True)
def jordan_centrality(parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Jordan centrality in ``O(n)``.

    Since ``parents[j] < j``, a reverse pass gives the subtree sizes ``sub`` of the
    tree rooted at 0, and ``psi(v) = max(max_c sub(c), n - sub(v))``.

    Args:
        parents (np.ndarray): Parent of each vertex with shape ``(n,)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Centrality ``psi`` and subtree sizes ``sub``,
            both with shape ``(n,)``.
    """
    n = len(parents)
    sub = np.ones(n, dtype=np.int64)
    biggest = np.zeros(n, dtype=np.int64)
    for j in range(n - 1, 0, -1):
        p = parents[j]
        sub[p] += sub[j]
        biggest[p] = max(biggest[p], sub[j])
    return np.maximum(biggest, n - sub), sub


def certified_k(psi: np.ndarray, sub: np.ndarray) -> int:
    """Computes the certificate of Lemma 2 of the note.

    With ``beta = 1 - psi(1) / n``, this is the smallest ``M`` such that
    ``|D(v)| <= (3/4) beta n`` for all ``v > M``. The lemma then guarantees that
    the root is in ``H_M`` whatever the tie-breaking.

    Args:
        psi (np.ndarray): Jordan centrality with shape ``(n,)``.
        sub (np.ndarray): Subtree sizes with shape ``(n,)``.

    Returns:
        int: The certified ``K``.
    """
    n = len(sub)
    beta = 1.0 - psi[0] / n
    violators = np.flatnonzero(sub[1:] > 0.75 * beta * n)
    return int(violators.max()) + 2 if violators.size else 1


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_theme(dark: bool) -> dict:
    """Returns matplotlib rc parameters blending in with the Streamlit theme."""
    fg = "#fafafa" if dark else "#31333f"
    return {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.edgecolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": fg,
        "grid.alpha": 0.12,
        "legend.frameon": False,
    }


def _edge_segments(points: np.ndarray, parents: np.ndarray, torus: bool) -> np.ndarray:
    """Builds the edge segments, drawn periodically on the torus.

    Each edge goes from the point to the closest image of its parent; if that
    image leaves ``[0, 1]^d``, the copy shifted back inside is added.
    """
    child = points[1:]
    head = points[parents[1:]]
    if torus:
        delta = child - head
        head = child - (delta - np.round(delta))
    segs = np.stack([child, head], axis=1)
    shift = (head < 0.0).astype(float) - (head > 1.0)
    wrapped = shift.any(axis=1)
    return np.concatenate([segs, segs[wrapped] + shift[wrapped, None]])


def draw_domain(
    ax,
    points: np.ndarray,
    parents: np.ndarray,
    psi: np.ndarray,
    H: np.ndarray,
    torus: bool,
    show_edges: bool,
    color_by_psi: bool,
):
    """Draws the points, the tree and the estimator in the chosen domain."""
    d = points.shape[1]
    kind = "Torus" if torus else "Cube"
    name = rf"$\mathbb{{T}}^{d}$" if torus else f"$[0,1]^{d}$"
    lim = (-0.02, 1.02)
    fg = plt.rcParams["text.color"]

    if d == 1 and torus:
        angle = 2.0 * np.pi * points[:, 0]
        xy = np.column_stack([np.cos(angle), np.sin(angle)])
        segs = np.stack([xy[1:], xy[parents[1:]]], axis=1)
        circle = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(np.cos(circle), np.sin(circle), color=fg, alpha=0.25, lw=1.0, zorder=0)
        ax.set_aspect("equal")
        ax.axis("off")
        title = f"{kind} {name} (circle)"
    elif d == 1:
        x = points[:, 0]
        t = np.linspace(0.0, 1.0, 12)
        h = 0.25 * np.abs(x[1:] - x[parents[1:]]) + 0.02
        xs = x[parents[1:], None] + t * (x[1:] - x[parents[1:]])[:, None]
        segs = np.stack([xs, 4.0 * h[:, None] * t * (1.0 - t)], axis=-1)
        xy = np.column_stack([x, np.zeros(len(x))])
        ax.plot([0, 1], [0, 0], color=fg, alpha=0.25, lw=1.2, zorder=0)
        ax.set(xlim=lim, ylim=(-0.03, max(0.15, 1.3 * h.max())))
        title = f"{kind} {name} (edges as arcs)"
    else:
        xy = points
        segs = _edge_segments(points, parents, torus)
        ax.set(xlim=lim, ylim=lim)
        if d == 2:
            ax.plot(
                [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=fg, alpha=0.25, lw=1.0, zorder=0
            )
            ax.set_aspect("equal")
            title = f"{kind} {name}"
        else:
            ax.set(zlim=lim, xlabel="x", ylabel="y", zlabel="z")
            title = f"{kind} {name} (3D view)"
    ax.set_title(title)

    if show_edges:
        collection = Line3DCollection if d == 3 else LineCollection
        ax.add_collection(collection(segs, colors=fg, alpha=0.3, lw=0.6, zorder=1))

    s = float(np.clip(4000.0 / len(points), 2.0, 35.0))
    style = {"c": psi, "cmap": "cool"} if color_by_psi else {"color": "steelblue"}
    sc = ax.scatter(*xy.T, s=s, label=None if color_by_psi else "points", **style)
    ax.scatter(
        *xy[H].T,
        s=2.7 * s + 199,
        facecolors="none",
        edgecolors=fg,
        linewidths=1.3,
        zorder=6,
        label=r"$H_K$",
    )
    ax.scatter(*xy[:1].T, s=154, marker="*", c="crimson", zorder=5, label="root 1")

    if color_by_psi:
        ax.figure.colorbar(sc, ax=ax, shrink=0.75, label=r"$\psi_n$")
    ax.legend(fontsize=12, loc="best")


def draw_psi(ax, psi: np.ndarray, K: int, kcert: int, R: int):
    """Draws the sorted psi values with the rank of vertex 1 and the bounds."""
    n = len(psi)
    beta = 1.0 - psi[0] / n
    ax.plot(
        np.arange(1, n + 1),
        np.sort(psi),
        lw=1.5,
        color="steelblue",
        label=r"sorted $\psi_n$",
    )
    ax.axhline(psi[0], color="crimson", ls="--", lw=1.2, label=r"$\psi_n(1)$")
    ax.axhline(
        n - 0.75 * beta * n,
        color="darkorange",
        ls=":",
        lw=1.2,
        label=r"$n-\frac{3}{4}\beta n$",
    )
    ax.plot([R], [psi[0]], "o", color="crimson", ms=6, zorder=5)
    ax.axvline(K, color="grey", ls="-.", lw=1.2, label=f"K = {K}")
    ax.axvline(
        kcert,
        color=plt.rcParams["text.color"],
        ls=":",
        lw=1.2,
        label=rf"$K_{{\rm cert}}$ = {kcert}",
    )
    ax.annotate(
        f"R = {R}",
        xy=(R, psi[0]),
        xytext=(0.10 * n, 0.5 * (psi[0] + psi.max())),
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        color="crimson",
    )
    ax.set(
        xlabel=r"vertices sorted by increasing $\psi_n$",
        ylabel=r"$\psi_n(v)$",
        title="Sorted Jordan centrality",
    )
    ax.legend(fontsize=12)


# ---------------------------------------------------------------------------
# Data (cached simulation)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def simulate_cached(domain: str, d: int, n: int, seed: int):
    """Simulates the points and the tree."""
    points = np.random.default_rng(seed).random((n, d))
    return points, simulate_nn_tree(points, domain == "torus")


@st.cache_data(show_spinner=False)
def analyse_cached(domain: str, d: int, n: int, seed: int, n0: int):
    """Computes the centrality of the tree of the first ``n0`` points."""
    points, parents = simulate_cached(domain, d, n, seed)
    psi, sub = jordan_centrality(parents[:n0])
    return points[:n0], parents[:n0], psi, sub


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NN tree root — torus or cube", layout="wide")
st.title("Recovering the root of the nearest neighbor tree")
st.markdown(
    "Points arrive i.i.d. uniform; each one connects to the closest of the points "
    "already present (Euclidean distance, periodic on the torus). "
    "The root is the first "
    "point, $1$. The algorithm ranks the vertices by **Jordan centrality** "
    "$\\psi_n(v)$ (size of the largest component after removing $v$) and estimates "
    "the root by $H_K$: the $K$ vertices with the smallest values."
)

if "seed" not in st.session_state:
    st.session_state.seed = 0


def _new_draw():
    st.session_state.seed = (st.session_state.seed + 1) % 10**9


with st.sidebar:
    st.header("Parameters")
    domain_label = st.radio(
        "Domain", [r"Torus $\mathbb{T}^d$", r"Cube $[0,1]^d$"], key="domain"
    )
    domain = "torus" if domain_label.startswith("Torus") else "cube"
    d = st.selectbox("Dimension $d$", [1, 2, 3], index=1, key="dim")
    n = st.slider("Number of sampled points $n$", 50, 30000, 2000, 50, key="n")
    K = st.slider("Size of the estimator $H_K$", 1, 100, 10, key="k")
    st.number_input("Seed", min_value=0, max_value=10**9, step=1, key="seed")
    st.button("🎲 New draw", on_click=_new_draw)

    with st.expander("Display and prefix"):
        show_edges = st.checkbox("Show edges", value=True, key="edges")
        color_psi = st.checkbox(r"Color points by $\psi_n$", key="color_psi")
        n0 = st.slider(
            "Points used for the computation (prefix)",
            20,
            n,
            n,
            10,
            key=f"n0_{n}",
        )

pts, par, psi, sub = analyse_cached(domain, d, n, st.session_state.seed, n0)

R = int(np.count_nonzero(psi <= psi[0]))
K_eff = min(K, n0)
H = np.argsort(psi, kind="stable")[:K_eff]
kcert = certified_k(psi, sub)

domain_txt = r"torus $\mathbb{T}^d$" if domain == "torus" else r"cube $[0,1]^d$"
st.caption(f"Domain: {domain_txt} · d = {d} · points used: {n0} (out of {n} sampled)")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Pessimistic rank $R$", R, help="number of vertices v such that ψ(v) ≤ ψ(1)"
)
col2.metric("Chosen K", K_eff)
col3.metric(
    "$K_{\\rm cert}$ (Lemma 2)",
    kcert,
    help="provable K for this tree, more conservative",
)
col4.metric("$\\psi_n(1)/n$", f"{psi[0] / n0:.2f}")

if K_eff >= R:
    st.success(
        f"✅ The root is **certified** to be in $H_K$: K = {K_eff} ≥ R = {R}, "
        "hence $1 \\in H_K$ whatever the tie-breaking."
    )
elif 0 in H:
    st.warning(
        f"⚠️ Vertex 1 is in the displayed $H_K$ (favorable tie-breaking), but "
        f"K = {K_eff} < R = {R}: no guarantee."
    )
else:
    st.error(
        f"❌ The root is not in $H_K$: K = {K_eff} < R = {R}. "
        "Success is guaranteed for K ≥ R on this tree."
    )

with plt.rc_context(plot_theme(st.context.theme.type == "dark")):
    fig = plt.figure(figsize=(13.5, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d" if d == 3 else None)
    draw_domain(ax1, pts, par, psi, H, domain == "torus", show_edges, color_psi)
    draw_psi(fig.add_subplot(1, 2, 2), psi, K_eff, kcert, R)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Estimator returned by the algorithm")
if H[0] == 0:
    st.markdown(
        "The vertex with the smallest centrality is **vertex 1: the root itself**."
    )
else:
    st.markdown(
        f"The vertex with the smallest centrality is {H[0] + 1} — the root "
        f"(vertex 1) is ranked {R} in the pessimistic order."
    )

st.dataframe(
    [
        {
            "vertex": int(v) + 1,
            "ψ": int(psi[v]),
            "ψ/n": round(float(psi[v]) / n0, 3),
            "is the root": "★" if v == 0 else "",
        }
        for v in H
    ],
    hide_index=True,
)

with st.expander("How does it work? (and algorithm code)"):
    st.markdown(
        r"""
**The model.** On the torus $\mathbb{T}^d=\mathbb{R}^d/\mathbb{Z}^d$ (Euclidean distance
modulo $\mathbb{Z}^d$) or on the cube $[0,1]^d$, the points $x_1,x_2,\dots$ are i.i.d.
uniform. For $j\ge 2$, $p(j)=\arg\min_{i<j} d(x_i,x_j)$, and $T_n$ has edges
$\{j,p(j)\}$.

**The statistic.** $\psi_n(v)$ = size of the largest connected component of
$T_n\setminus\{v\}$. It depends only on the unlabeled graph and is computed in $O(n)$
(subtree sizes, arbitrary rooting). The estimator is
$H_K$ = the $K$ smallest values.

**The pessimistic rank.** $R=\#\{v:\psi_n(v)\le\psi_n(1)\}$. If $K\ge R$, the root is
in $H_K$ for *any* tie-breaking; this is exactly what the table above measures and
what the "prefix" slider tracks (the trees are nested: the prefix is the tree of the
first $n_0$ points).

**Implementation.** Exact simulation of $T_n$ in $\tilde O(n)$: batched nearest neighbor
queries in a KD-tree (`boxsize=1` for the torus), with refinement of $k$. $\psi_n$ is
computed from the parents alone, without geometry.
"""
    )
    st.code(inspect.getsource(jordan_centrality.py_func), language="python")
