"""
2D visualization: how the standard Mondrian partitions the space vs the CART-driven Mondrian variants.

Data: X ~ Unif[0,1]^2, binary label y in {0,1} (fit as regression on 0/1, so SSE = Gini-type impurity).
Three toy problems:
  stripes : y = 1[sin(4*pi*x1) > 0]          -> only x1 matters; adaptive cuts should be mostly vertical
  disk    : y = 1[point inside a disk]        -> adaptive cuts should concentrate around the boundary
  xor     : y = 1[(x1 > .5) xor (x2 > .5)]    -> a hard case for any score built from single splits:
                                                 no single axis split reduces SSE in expectation

Fair comparison: the two methods' lifetimes are on different scales (width vs SSE units), so instead of
fixing a lifetime we grow every tree fully (lifetime = inf) and truncate it at its K-th split time, so
every tree has the same number of leaves (--leaves). Split times increase along each path, so truncating
by time always gives a valid subtree, i.e. the partition the tree had when it reached K leaves.
(A CART tree can have fewer leaves than K: it stops splitting once every cell is pure / all scores are 0.)

Outputs (results/):
  partition2d_trees.png   one tree per method, its cells shaded by the cell mean, with the data
  partition2d_forest.png  forest prediction (mean over --trees trees, each truncated to K leaves)
Also prints, per dataset and method: % of cuts on x1, mean leaves, forest test accuracy (threshold 0.5).

Run:
  ~/miniconda3/envs/adaptive_mondrian/bin/python viz_partition_2d.py
  ~/miniconda3/envs/adaptive_mondrian/bin/python viz_partition_2d.py --leaves 25 --n 800 --noise 0.05
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from adaptive_mondrian import AdaptiveMondrianForestRegressor

BETA = 10  # same as compare.py / diagnostic.py
METHODS = [  # (title, score, beta)
    ("Mondrian", "width", None),
    ("Mondrian-CART", "cart", None),
    ("Mondrian-CART-null", "cart_null", None),
    (f"Mondrian-CART-exp (β={BETA})", "cart", BETA),
]

# diverging blue <-> red with neutral gray midpoint: 0 = blue, 1 = red, 0.5 = "no information"
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"
CMAP = LinearSegmentedColormap.from_list("bgr", [BLUE, MID, RED])
INK, MUTED = "#2b2b2b", "#6b6b6b"


# ---------------------------------------------------------------- data
def label_fn(name):
    if name == "stripes":
        return lambda X: (np.sin(4 * np.pi * X[:, 0]) > 0).astype(float)
    if name == "disk":
        return lambda X: (((X[:, 0] - 0.65) ** 2 + (X[:, 1] - 0.6) ** 2) < 0.22 ** 2).astype(float)
    if name == "xor":
        return lambda X: ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)
    raise ValueError(name)


def make_data(name, n, noise, rng):
    X = rng.rand(n, 2)
    y = label_fn(name)(X)
    flip = rng.rand(n) < noise  # label noise
    y[flip] = 1 - y[flip]
    return X, y


# ---------------------------------------------------------------- tree helpers
def cutoff_for_leaves(tree, k):
    """Lifetime that keeps the first k-1 splits (-> k leaves). inf if the tree has fewer splits."""
    times = np.sort(tree["time"][tree["feature"] >= 0])
    return times[k - 1] if len(times) >= k else np.inf


def cells(tree, lifetime, box=((0.0, 1.0), (0.0, 1.0))):
    """Leaf rectangles of the tree truncated at `lifetime`: list of (x0, x1, y0, y1, value)."""
    out, stack = [], [(0, [list(box[0]), list(box[1])])]
    used = []  # features of the kept splits
    while stack:
        node, b = stack.pop()
        j = tree["feature"][node]
        if j < 0 or tree["time"][node] >= lifetime:
            out.append((b[0][0], b[0][1], b[1][0], b[1][1], tree["value"][node]))
            continue
        used.append(j)
        t = tree["thr"][node]
        bl = [b[0][:], b[1][:]]; bl[j][1] = t
        br = [b[0][:], b[1][:]]; br[j][0] = t
        stack.append((tree["left"][node], bl))
        stack.append((tree["right"][node], br))
    return out, np.array(used)


def predict_tree(tree, X, lifetime):
    out = np.empty(len(X))
    stack = [(0, np.arange(len(X)))]
    while stack:
        node, idx = stack.pop()
        if len(idx) == 0:
            continue
        j = tree["feature"][node]
        if j < 0 or tree["time"][node] >= lifetime:
            out[idx] = tree["value"][node]
            continue
        m = X[idx, j] <= tree["thr"][node]
        stack.append((tree["left"][node], idx[m]))
        stack.append((tree["right"][node], idx[~m]))
    return out


# ---------------------------------------------------------------- plotting
def style(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8, colors=MUTED, length=2)
    for s in ax.spines.values():
        s.set_color("#cfcfcf")


def scatter(ax, X, y, size=6):
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y > 0.5, RED, BLUE), s=size, linewidths=0.3,
               edgecolors="white", zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="stripes,disk,xor")
    ap.add_argument("--n", type=int, default=600, help="training points")
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--leaves", type=int, default=40, help="leaves per tree (matched across methods)")
    ap.add_argument("--trees", type=int, default=50, help="trees in the forest figure")
    ap.add_argument("--noise", type=float, default=0.0, help="label flip probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/partition2d")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    nc = len(METHODS) + 1
    fig_t, ax_t = plt.subplots(len(datasets), nc, figsize=(3.0 * nc, 3.1 * len(datasets)), squeeze=False)
    fig_f, ax_f = plt.subplots(len(datasets), nc, figsize=(3.0 * nc, 3.1 * len(datasets)), squeeze=False)

    g = np.linspace(0, 1, 200)
    G = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])

    for r, name in enumerate(datasets):
        rng = np.random.RandomState(args.seed)
        X, y = make_data(name, args.n, args.noise, rng)
        Xte = rng.rand(args.n_test, 2)
        yte = label_fn(name)(Xte)  # clean test labels
        print(f"\n=== {name}: n={args.n}, leaves/tree={args.leaves}, trees={args.trees}, noise={args.noise}")
        print(f"{'method':32s} {'% cuts on x1':>13s} {'mean leaves':>12s} {'forest test acc':>16s}")

        for a in (ax_t[r, 0], ax_f[r, 0]):  # column 0: the data / true labels
            a.imshow(label_fn(name)(G).reshape(200, 200), origin="lower", extent=(0, 1, 0, 1),
                     cmap=CMAP, vmin=0, vmax=1, alpha=0.25)
            scatter(a, X, y)
            style(a)
            a.set_ylabel(name, fontsize=11, color=INK)
        ax_t[r, 0].set_title("data" if r == 0 else "", fontsize=10, color=INK)
        ax_f[r, 0].set_title("data" if r == 0 else "", fontsize=10, color=INK)

        for c, (title, score, beta) in enumerate(METHODS, start=1):
            f = AdaptiveMondrianForestRegressor(args.trees, np.inf, score=score, random_state=args.seed,
                                                beta=beta).fit(X, y)
            cuts = [cutoff_for_leaves(t, args.leaves) for t in f.trees_]
            used_all, n_leaves = [], []
            for t, L in zip(f.trees_, cuts):
                cl, used = cells(t, L)
                used_all.append(used); n_leaves.append(len(cl))
            used_all = np.concatenate(used_all)
            pct_x1 = 100 * np.mean(used_all == 0) if len(used_all) else np.nan
            pred_te = np.mean([predict_tree(t, Xte, L) for t, L in zip(f.trees_, cuts)], axis=0)
            acc = np.mean((pred_te > 0.5) == (yte > 0.5))
            print(f"{title:32s} {pct_x1:12.1f}% {np.mean(n_leaves):12.1f} {acc:16.3f}")

            # --- single tree (the first one)
            a = ax_t[r, c]
            cl, used = cells(f.trees_[0], cuts[0])
            for x0, x1, y0, y1, v in cl:
                a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=CMAP(v), alpha=0.55,
                                      edgecolor=INK, linewidth=0.6, zorder=2))
            scatter(a, X, y, size=3)
            style(a)
            cut_txt = f"x1 cuts {100*np.mean(used==0):.0f}%" if len(used) else "no cuts"
            a.set_title((title + "\n" if r == 0 else "") + f"{len(cl)} {'leaf' if len(cl) == 1 else 'leaves'}, {cut_txt}",
                        fontsize=9, color=INK)

            # --- forest prediction
            a = ax_f[r, c]
            P = np.mean([predict_tree(t, G, L) for t, L in zip(f.trees_, cuts)], axis=0)
            a.imshow(P.reshape(200, 200), origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmin=0, vmax=1)
            a.contour(g, g, P.reshape(200, 200), levels=[0.5], colors=INK, linewidths=0.8)
            style(a)
            a.set_title((title + "\n" if r == 0 else "") + f"test acc {acc:.3f}", fontsize=9, color=INK)

    fig_t.suptitle(f"One tree per method, {args.leaves} leaves each (cell color = cell mean of y)",
                   fontsize=12, color=INK)
    fig_f.suptitle(f"Forest prediction ({args.trees} trees × {args.leaves} leaves); black line = 0.5 boundary",
                   fontsize=12, color=INK)
    for fig, suffix in ((fig_t, "trees"), (fig_f, "forest")):
        fig.tight_layout()
        path = f"{args.out}_{suffix}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
