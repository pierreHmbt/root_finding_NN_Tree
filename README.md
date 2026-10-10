# Finding Adam — recovering the root of a random nearest-neighbor tree

An interactive [Streamlit](https://streamlit.io) app that **recovers the root of a random
nearest-neighbor tree from the graph alone**. Points arrive i.i.d. uniform; each new point
connects to its nearest predecessor. The first point ("Adam") is the root, and the goal is
to find it **without any labels and without using the geometry** — from the unlabeled tree.

> 🔗 **Live app:** <https://root-random-nn-tree.streamlit.app/>

---

## The model

On the torus $\mathbb{T}^d = \mathbb{R}^d / \mathbb{Z}^d$ (periodic Euclidean distance) or on
the cube $[0,1]^d$, the points $x_1, x_2, \dots$ are i.i.d. uniform. Each point connects to the
closest of the points already present:

$$p(j) = \arg\min_{i \lt j} d(x_i, x_j),
\qquad
T_n = \bigl([n], \lbrace\lbrace j, p(j)\rbrace\rbrace_{2 \le j \le n}\bigr).$$

The root is the first point, $1$.

## The statistic

The **Jordan centrality** of a vertex is the size of the largest component obtained by
deleting it:

$$\psi_n(v) = \max\bigl\lbrace|C| : C \text{ is a connected component of } T_n \setminus \lbrace v\rbrace\bigr\rbrace.$$

The estimator is $H_K$ = the $K$ vertices with the smallest values of $\psi_n$ (ties broken
arbitrarily).

**Theorem.** For every $d \ge 1$ and every $\varepsilon \in (0,1)$, there is
$K = K(d, \varepsilon)$ — **independent of $n$** — such that

$$\liminf_{n \to \infty} \mathbb{P}\bigl(1 \in H_K\bigr) \ge 1 - \varepsilon .$$

The same proof works on the cube.

---

## What the app shows

* The tree itself on the chosen domain: the true root (red star) and the estimator $H_K$
  (black circles); the best root candidate is highlighted.
* A **verdict**:
  * ✅ **certified** as soon as $K \ge R$, where $R = \lvert\lbrace v : \psi_n(v) \le \psi_n(1)\rbrace\rvert$ is the
    pessimistic rank — then $1 \in H_K$ for *every* tie-breaking;
  * ⚠️ the root happens to be in the displayed $H_K$ (favorable tie-breaking) but there is no
    guarantee;
  * ❌ the root is not in $H_K$.
* Key metrics: $R$, the chosen $K$, the per-tree certificate $K_{\rm cert}$, and $\psi_n(1)/n$.
* The table of the vertices in $H_K$ with their $\psi_n$ values.
* The sorted values of $\psi_n$, with the level $\psi_n(1)$ and the markers $R$, $K$,
  $K_{\rm cert}$.

### Sidebar parameters

| Parameter | Range / options |
|---|---|
| **Domain** | torus $\mathbb{T}^d$ (periodic distance) or cube $[0,1]^d$ |
| **Dimension** $d$ | $1, 2, 3$ (circle for the torus $d=1$, arcs for the cube $d=1$, 3D view for $d=3$) |
| **Number of points** $n$ | 50 – 30 000 |
| **Estimator size** $K$ | 1 – 100 |
| **Seed** + 🎲 **New draw** | reproducible random draws |
| **Display options** | show edges, color vertices by $\psi_n$ |
| **Prefix** $n_0$ | compute the estimator on the first $n_0$ points (nested trees) |

The app is **fully interactive** thanks to Streamlit caching, even at $n = 30{,}000$.

---

## Installation

Python dependencies: `streamlit`, `numpy`, `scipy`, `matplotlib`, `pandas`
(see [`requirements.txt`](requirements.txt)).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app_root_finding_nn_tree.py
```

It opens at <http://localhost:8501>.

> Or use the hosted version: <https://root-random-nn-tree.streamlit.app/>

---

## Choosing $K$

**In theory**, no closed form is known: $K(d, \varepsilon)$ is existential, and the correct
order in $1/\varepsilon$ is an open question. The note gives an upper bound
$K(\varepsilon) \le \varepsilon^{-C_d}$ (non-explicit exponent) and a polylogarithmic lower
bound in $1/\varepsilon$; as $d \to \infty$ (uniform attachment limit),
$K \sim \exp(\Theta(\sqrt{\log(1/\varepsilon)}))$.

**In practice**, calibrate by Monte-Carlo: simulate $m$ trees, compute $R$ on each, and take
$K = \widehat q_{1-\varepsilon}(R)$ (the empirical quantile). This is legitimate because the
law of $R$ does not drift with $n$ — calibrate once for a given $(d, \text{domain})$, then keep
$K$ fixed. Allow a margin for sampling fluctuations (binomial or DKW bound).

For $d=2$ on the torus: $K \approx 6$ at a 10% error level, $K \approx 20$ at 1%.

**Two useful quantities**, both shown in the app:

* $R = \lvert\lbrace v : \psi_n(v) \le \psi_n(1)\rbrace\rvert$ — the **pessimistic rank**: the root lies in $H_K$
  for every tie-breaking **iff** $K \ge R$;
* $K_{\rm cert}$ — the **certificate of Lemma 2**: the smallest *provable, tree-by-tree* $K$.
  It uses $\psi_n(1)$ and the descendant sets, hence the identity of the root; it is an analysis
  tool (an upper bound on $K(d,\varepsilon)$ via its quantiles), not an output of the algorithm.
  For $d=2$: median $\approx 4$, $q_{90} \approx 10$–$17$, $q_{99} \approx 40$–$70$.

---

## Implementation & correctness

* The **simulation is exact** and fast: batched $k$-nearest-neighbor search in a periodic KD-tree
  (`boxsize=1` on the torus) with doubling of $k$, in $\tilde O(n)$ — under a second for
  $n = 2^{16}$.
* $\psi_n$ is computed in $O(n)$ **on the unlabeled graph only** (subtree sizes from an
  arbitrary rooting), with no geometry involved.
* `simulate_nn_tree` is checked against an $O(n^2)$ brute force in dimensions 1–3, and
  `jordan_centrality` is checked against vertex-deletion brute force and against relabeling
  invariance.
* The app is **self-contained** (the algorithm functions are deliberately duplicated from the
  notebook) and relies on Streamlit caching to stay interactive.

---

## References

* L. Lichev, D. Mitsche. _New results for the random nearest neighbor tree._ Probab. Theory
  Related Fields 189:229–279, 2024.
* A. Brandenberger, C. Marcussen, E. Mossel, M. Sudan. _Finding the root in random nearest
  neighbor trees._ Random Structures & Algorithms 68(2):e70057, 2026.
* S. Bubeck, L. Devroye, G. Lugosi. _Finding Adam in random growing trees._ Random Structures &
  Algorithms 50(2):158–172, 2017.
* D. Aldous. _Random partitions of the plane via Poissonian coloring…_ Ann. Probab.
  46(4):2000–2037, 2018.
* J. Preater. _A species of voter model driven by immigration._ Statist. Probab. Lett.
  79(20):2131–2137, 2009.
* A.-L. Basdevant, G. Blanc, N. Curien, A. Singh. _Fractal properties of the frontier in
  Poissonian coloring._ ALEA 20:1565–1586, 2023.
