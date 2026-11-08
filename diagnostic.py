"""
Diagnostic: which features do the splits land on?

For the standard Mondrian (width), the data-adaptive Mondrian (CART score), its null-corrected version, its
exponential-feature-choice version (beta=BETA, see adaptive_mondrian.py) and a sklearn random forest
(default settings), count the fraction of splits (internal nodes) made on the 3 relevant features
(columns 0,1,2), overall and by the depth of the node in the tree. Splits on the other d-3 columns
are wasted.

How to read it:
  - A method that ignores the data puts about 3/d of its splits on the relevant features.
  - A greedy method (RF) should put most of its early splits there.
  - Question: where does the CART variant sit, and does it fall off at larger depth (small cells,
    where the signal is weak and chance reductions on noise features compete with it)?
Deeper nodes contain fewer points, so the share on relevant features is usually lower there.

Lifetimes of the two Mondrian forests are chosen on a validation set exactly as in compare.py
(same grids, 10 trees), so the trees inspected are the ones that would be used for prediction.
Counts are pooled over trees and seeds.

Usage:  python diagnostic.py                 # 5 seeds, d in 10,20,30,50
        python diagnostic.py 3 20,50         # 3 seeds, d in 20,50
Needs adaptive_mondrian.py in the same folder.
"""
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from adaptive_mondrian import AdaptiveMondrianForestRegressor

N_TR_LIST = [3200, 6400]  # training sizes, as in compare.py; each is a separate run
N_VAL = 400
RELEVANT = (0, 1, 2)
M = 10
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DIMS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 20, 30, 50]

WIDTH_LIFETIMES = (0.5, 1000, 12)       # grids as in compare.py (extended for n_train > 800)
CART_LIFETIMES_D10 = (0.1, 5000, 12)    # multiplied by D/10, as in compare.py
BINS = [(0, 2), (3, 5), (6, 9), (10, 10**9)]
BIN_NAMES = ["depth 0-2", "depth 3-5", "depth 6-9", "depth 10+"]
BETA = 10  # as in compare.py
METHODS = ["Mondrian", "Mondrian-CART", "Mondrian-CART-null", "Mondrian-CART-exp", "RF"]

mse = lambda a, b: float(np.mean((a - b) ** 2))


def make(rng, n, D):
    X = rng.rand(n, D)
    y = np.sin(4 * X[:, 0]) + 2 * (X[:, 1] - 0.5) ** 2 + X[:, 2] + 0.1 * rng.randn(n)
    return X, y


def lifetime_grid(lo, hi, n_pts, N):
    """Same as compare.lifetime_grid: the n_train = 800 grid, extended upwards to hi * N/800."""
    step = (hi / lo) ** (1.0 / (n_pts - 1))
    extra = max(0, int(np.ceil(np.log(N / 800.0) / np.log(step) - 1e-9)))
    return lo * step ** np.arange(n_pts + extra)


def mondrian_splits(tree, lifetime=np.inf):
    """(depth, feature) for every internal node of a tree from AdaptiveMondrianForestRegressor,
    truncated at `lifetime` (nodes split at a time >= lifetime are leaves)."""
    out, stack = [], [(0, 0)]
    while stack:
        node, depth = stack.pop()
        j = tree["feature"][node]
        if j < 0 or tree["time"][node] >= lifetime:  # leaf
            continue
        out.append((depth, int(j)))
        stack.append((tree["left"][node], depth + 1))
        stack.append((tree["right"][node], depth + 1))
    return out


def sklearn_splits(est):
    """(depth, feature) for every internal node of a fitted sklearn decision tree."""
    t = est.tree_
    out, stack = [], [(0, 0)]
    while stack:
        node, depth = stack.pop()
        j = t.feature[node]
        if j < 0:  # leaf (sklearn marks leaves with -2)
            continue
        out.append((depth, int(j)))
        stack.append((t.children_left[node], depth + 1))
        stack.append((t.children_right[node], depth + 1))
    return out


def count(splits):
    """Per depth bin: number of splits on relevant features, number of splits in total."""
    rel, tot = np.zeros(len(BINS)), np.zeros(len(BINS))
    for depth, j in splits:
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= depth <= hi)
        tot[b] += 1
        rel[b] += int(j in RELEVANT)
    return rel, tot


def run_seed(seed, D, N_TR):
    t0 = time.time()
    rng = np.random.RandomState(seed)
    Xtr, ytr = make(rng, N_TR, D)
    Xva, yva = make(rng, N_VAL, D)
    out = {}

    width_lams = lifetime_grid(*WIDTH_LIFETIMES, N_TR)
    cart_lams = lifetime_grid(*CART_LIFETIMES_D10, N_TR) * (D / 10.0)
    null_lams = np.append(cart_lams, np.inf)
    for name, score, lams, beta in [("Mondrian", "width", width_lams, None),
                                    ("Mondrian-CART", "cart", cart_lams, None),
                                    ("Mondrian-CART-null", "cart_null", null_lams, None),
                                    ("Mondrian-CART-exp", "cart", cart_lams, BETA)]:
        # nested lifetimes: grow once at the largest lifetime, truncate for the others (as in compare.py)
        f = AdaptiveMondrianForestRegressor(M, max(lams), score=score, random_state=seed, beta=beta).fit(Xtr, ytr)
        best = None
        for lam in lams:
            v = mse(f.predict(Xva, lam), yva)
            if best is None or v < best[0]:
                best = (v, lam)
        splits = [s for t in f.trees_ for s in mondrian_splits(t, best[1])]
        out[name] = (count(splits), len(splits) / M, best[1])

    rf = RandomForestRegressor(M, random_state=seed).fit(Xtr, ytr)
    splits = [s for e in rf.estimators_ for s in sklearn_splits(e)]
    out["RF"] = (count(splits), len(splits) / M, None)

    print(f"  n={N_TR} d={D} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return out


def report(D, N_TR, results):
    print(f"\n=== n = {N_TR}, d = {D}: share of splits on the 3 relevant features "
          f"(feature-blind chance level = {3 / D:.1%}), {len(results)} seeds ===")
    print(f"{'method':20s}" + "".join(f"{b:>12s}" for b in BIN_NAMES)
          + f"{'all splits':>12s}{'splits/tree':>13s}{'median lifetime':>17s}")
    for name in METHODS:
        rel = sum(r[name][0][0] for r in results)
        tot = sum(r[name][0][1] for r in results)
        cells = "".join(f"{rel[b] / tot[b]:12.1%}" if tot[b] > 0 else f"{'-':>12s}" for b in range(len(BINS)))
        overall = rel.sum() / tot.sum()
        per_tree = np.mean([r[name][1] for r in results])
        lams = [r[name][2] for r in results]
        life = f"{np.median(lams):.3g}" if lams[0] is not None else "-"
        print(f"{name:20s}{cells}{overall:12.1%}{per_tree:13.0f}{life:>17s}")


if __name__ == "__main__":
    for N_TR in N_TR_LIST:
        for D in DIMS:
            print(f"\n##### n = {N_TR}, d = {D}: running {N_SEEDS} seeds, {M} trees", flush=True)
            results = Parallel(n_jobs=-1)(delayed(run_seed)(s, D, N_TR) for s in range(N_SEEDS))
            report(D, N_TR, results)