"""Streamlit app — recovering the root of the nearest neighbor tree.

Choose the domain (torus T^d or cube [0,1]^d), the dimension d and the number
of sampled points n; the app then simulates the tree T_n, computes the Jordan
centrality psi_n on the unlabeled graph and displays the root estimator H_K:
the K vertices with the smallest values of psi_n.

Run with:
    ~/.venv/bin/streamlit run app_root_finding_nn_tree.py
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
# The algorithm
# ---------------------------------------------------------------------------


def simulate_nn_tree(points, torus=True, k0=8, workers=-1):
    """Online nearest neighbor tree.

    parents[j] = argmin_{i<j} d(x_i, x_j); periodic Euclidean distance
    (``boxsize=1``) on the torus, ordinary distance on the cube. Exact method:
    batched kNN queries, doubling k for the points with no predecessor in
    the list.
    """
    n = len(points)
    parents = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return parents
    tree = cKDTree(points, boxsize=1.0 if torus else None)
    todo = np.arange(1, n)
    k = min(k0, n - 1)
    while todo.size:
        kq = min(k + 1, n)  # +1: the point itself, at distance zero
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
    """psi(v) = size of the largest component of T - v, in O(n).

    Roots the tree at 0 (any root gives the same psi) and computes the
    subtree sizes sub: psi(v) = max(max_c sub(c), n - sub(v)).
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
    """Certificate from Lemma 2 of the note: smallest M such that, with
    beta = 1 - psi(1)/n, |D(v)| <= (3/4) beta n for all v > M.
    The lemma then guarantees rank(root) <= M, hence 1 in H_M whatever the
    tie-breaking.
    """
    n = sub.size
    beta = 1.0 - psi[0] / n
    if beta <= 0.0:
        return n
    # vertices v >= 2 (1-indexed) violating (B): |D(v)| > (3/4) beta n
    violators = np.flatnonzero(sub[1:] > 0.75 * beta * n) + 1   # 0-indexed labels
    return int(violators.max()) + 1 if violators.size else 1   # M = largest violating v


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _point_size(n):
    return float(np.clip(4000.0 / max(n, 1), 2.0, 35.0))


def _edge_segments(xy, parents, torus):
    """Edge segments, drawn periodically on the torus.

    Each edge goes from the point to the closest image of its parent; if that
    image lies outside the cube [0,1]^d, the copy shifted by ±1 (per
    coordinate) is added: the edge crosses the boundary and reappears on the
    other side.
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
    """Scatter plot (colored by psi if requested) + root and H_K highlighted."""
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
            zorder=5, depthshade=False, label="root 1",
        )
    else:
        ax.scatter(
            [root[0]], [root[1]], s=200, marker="*", c="crimson", zorder=5, label="root 1"
        )
    return sc


def draw_domain(ax, points, parents, psi, H, torus, d, show_edges, color_by_psi):
    """Draws the points, the tree and the estimator in the chosen domain."""
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
        ax.set_title(r"Torus $\mathbb{T}^1$ (circle)")

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
            r"Torus $\mathbb{T}^2$ (shortest edges)" if torus else r"Cube $[0,1]^2$"
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
            r"Torus $\mathbb{T}^3$ (3D view)" if torus else r"Cube $[0,1]^3$ (3D view)"
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
        ax.set_title(r"Cube $[0,1]$ (edges as arcs)")

    if sc is not None:
        ax.figure.colorbar(sc, ax=ax, shrink=0.75, label=r"$\psi_n$")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="best")
    return ax


def draw_psi(ax, psi, K, kcert, R):
    """Sorted psi values: rank of vertex 1 and bounds of the algorithm."""
    n = len(psi)
    order = np.argsort(psi, kind="stable")
    ax.plot(np.arange(1, n + 1), psi[order], lw=1.5, color="steelblue", label=r"sorted $\psi_n$")
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
        xlabel=r"vertices sorted by increasing $\psi_n$",
        ylabel=r"$\psi_n(v)$",
        title="Sorted Jordan centrality",
    )
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Data (cached simulation)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def simulate_cached(domain, d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, int(d)))
    parents = simulate_nn_tree(points, torus=(domain == "torus"))
    return points, parents


@st.cache_data(show_spinner=False)
def analyse_cached(domain, d, n, seed, n0):
    points, parents = simulate_cached(domain, d, n, seed)
    pts = points[:n0].copy()
    par = parents[:n0].copy()
    edges = np.column_stack([np.arange(1, n0), par[1:]])
    t0 = time.perf_counter()
    psi, sub = jordan_centrality(n0, edges, return_subtree=True)
    dt = time.perf_counter() - t0
    return pts, par, psi, sub, dt


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NN tree root — torus or cube", layout="wide")
st.title("Recovering the root of the nearest neighbor tree")
st.markdown(
    "Points arrive i.i.d. uniform; each one connects to the closest of the points "
    "already present (Euclidean distance, periodic on the torus). The root is the first "
    "point, $1$. The algorithm ranks the vertices by **Jordan centrality** "
    "$\\psi_n(v)$ (size of the largest component after removing $v$) and estimates "
    "the root by $H_K$: the $K$ vertices with the smallest values."
)

if "seed" not in st.session_state:
    st.session_state.seed = 0


def _new_draw():
    st.session_state.seed = (int(st.session_state.seed) + 1) % 10**9


with st.sidebar:
    st.header("Parameters")
    domain_label = st.radio(
        "Domain", [r"Torus $\mathbb{T}^d$", r"Cube $[0,1]^d$"], index=0, key="domain"
    )
    domain = "torus" if domain_label.startswith("Torus") else "cube"
    d = st.selectbox("Dimension $d$", [1, 2, 3], index=1, key="dim")
    n = st.slider(
        "Number of sampled points $n$",
        min_value=50,
        max_value=30000,
        value=2000,
        step=50,
        key="n",
    )
    K = st.slider("Size of the estimator $H_K$", 1, 100, 10, key="k")
    st.number_input("Seed", min_value=0, max_value=10**9, step=1, key="seed")
    st.button("🎲 New draw", on_click=_new_draw)

    with st.expander("Display and prefix"):
        show_edges = st.checkbox("Show edges", value=True, key="edges")
        color_psi = st.checkbox(r"Color points by $\psi_n$", value=False, key="color_psi")
        n0 = st.slider(
            "Points used for the computation (prefix)",
            min_value=20,
            max_value=n,
            value=n,
            step=10,
            key=f"n0_{n}",
        )

with st.spinner("Simulating the tree and computing $\\psi_n$..."):
    pts, par, psi, sub, dt_psi = analyse_cached(domain, d, n, st.session_state.seed, n0)

n0 = len(pts)
R = int(np.count_nonzero(psi <= psi[0]))
K_eff = int(min(K, n0))
H = np.argsort(psi, kind="stable")[:K_eff]
root_in_H = bool(0 in H)
kcert = certified_k(psi, sub)
v_min = int(np.argmin(psi))

domain_txt = r"torus $\mathbb{T}^d$" if domain == "torus" else r"cube $[0,1]^d$"
st.caption(
    f"Domain: {domain_txt} · d = {d} · points used: {n0} (out of {n} sampled) · "
    f"$\\psi_n$ computation: {dt_psi * 1000:.0f} ms"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pessimistic rank $R$", R, help="number of vertices v such that ψ(v) ≤ ψ(1)")
col2.metric("Chosen K", K_eff)
col3.metric("$K_{\\rm cert}$ (Lemma 2)", kcert, help="provable K for this tree, more conservative")
col4.metric("$\\psi_n(1)/n$", f"{psi[0] / n0:.2f}")

if K_eff >= R:
    st.success(
        f"✅ The root is **certified** to be in $H_K$: K = {K_eff} ≥ R = {R}, "
        "hence $1 \\in H_K$ whatever the tie-breaking."
    )
elif root_in_H:
    st.warning(
        f"⚠️ Vertex 1 is in the displayed $H_K$ (favorable tie-breaking), but "
        f"K = {K_eff} < R = {R}: no guarantee."
    )
else:
    st.error(
        f"❌ The root is not in $H_K$: K = {K_eff} < R = {R}. "
        "Success is guaranteed for K ≥ R on this tree."
    )

fig = plt.figure(figsize=(13.5, 5.4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d") if d == 3 else fig.add_subplot(1, 2, 1)
draw_domain(ax1, pts, par, psi, H, domain == "torus", d, show_edges, color_psi)
ax2 = fig.add_subplot(1, 2, 2)
draw_psi(ax2, psi, K_eff, kcert, R)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Estimator returned by the algorithm")
if v_min == 0:
    st.markdown("The vertex with the smallest centrality is **vertex 1: the root itself**.")
else:
    st.markdown(
        f"The vertex with the smallest centrality is {v_min + 1} — the root (vertex 1) is "
        f"ranked {R} in the pessimistic order."
    )

rows = [
    {
        "vertex": int(v) + 1,
        "ψ": int(psi[v]),
        "ψ/n": round(float(psi[v]) / n0, 3),
        "is the root": "★" if v == 0 else "",
    }
    for v in H
]
st.dataframe(pd.DataFrame(rows), hide_index=True)

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
computed from the edges alone, without geometry.
"""
    )
    st.code(
        """def jordan_centrality(n, edges, return_subtree=False):
    ...
    sub = np.ones(n, dtype=np.int64)
    for x in order[:0:-1]:          # reverse breadth-first order
        sub[parent[x]] += sub[x]
    psi = np.maximum(maxchild, n - sub)
    return (psi, sub) if return_subtree else psi""",
        language="python",
    )
