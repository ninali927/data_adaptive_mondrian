"""
Comparison on data where y depends on only 3 latent coordinates u_0,u_1,u_2 out of d inputs.
X is always Uniform[0,1]^d and y = sin(4 u_0) + 2 (u_1 - 0.5)^2 + u_2 + noise. d is swept over DIMS.
  default (axis-aligned): u_k = column k of X, so columns 0,1,2 matter and the other d-3 are pure noise.
  "rotated" mode:         u_k = 0.5 + (x - 0.5) . b_k with b_0,b_1,b_2 random orthonormal directions in R^d
                          (one fresh draw per seed), so every column carries a small part of the signal and
                          no single column is irrelevant. Only the target is rotated, not X.

Methods
  Mondrian             standard Mondrian forest (score = box width)
  Mondrian-CART        the data-adaptive variant (score = expected CART reduction), feature p_j ∝ s_j
  Mondrian-CART-null   variant of Mondrian-CART with the noise baseline SSE_parent/(n-1) subtracted from
                       each feature's score (clipped at 0): score="cart_null". Feature p_j ∝ s_j.
  Mondrian-CART-exp    same score as Mondrian-CART (no null correction), but the feature is drawn with
                       p_j ∝ exp(BETA * s_j / max_k s_k) instead of p_j ∝ s_j (beta=BETA in
                       AdaptiveMondrianForestRegressor). CART-null and CART-exp are two separate fixes of
                       Mondrian-CART, compared side by side.
  RF                   sklearn RandomForestRegressor with max_features and min_samples_leaf tuned on
                       validation (the Mondrian-type forests get their lifetime tuned the same way).
Main comparison: Mondrian vs the adaptive variants vs RF. The summary also reports how much of the gap
between Mondrian and RF each adaptive variant closes.

Protocol
  - Every training size in N_TR_LIST is a separate run (all d, all seeds), reported separately.
  - N_SEEDS independent datasets per d (train/val/test regenerated per seed).
  - Mondrian-type forests: lifetime picked on the validation set from a geometric grid,
    separately for each number of trees M. The M=50 forest is grown once, with the largest lifetime
    of the grid; smaller lifetimes truncate its trees at that lifetime, and the first 10 / 30 trees
    are used for M=10 / 30 (nested lifetimes and nested forests, same randomness).
    The CART lifetime grid is multiplied by d/10, because the exponential rate is
    (sum of scores)/d and only 3 features have large scores. Both grids are extended upwards for
    n_train > 800 (see lifetime_grid), because larger samples can support deeper trees.
  - RF: 9 settings of (max_features, min_samples_leaf), picked on validation, per M.
  - Reported: mean +- std of TEST mse over seeds, the chosen setting, and for Mondrian-type
    forests how often the chosen lifetime sat at the low / high edge of the grid. A "high" edge pick
    usually just means "fully grown trees" (for cart_null: trees that kept growing until no feature beat
    chance), and then a wider grid would change nothing.

Usage:  python compare.py                  # 10 seeds, d in 10,20,30,50
        python compare.py 2                # quick run with 2 seeds
        python compare.py 10 20,30,50      # choose the dimensions
        python compare.py 10 10,20,30,50 rotated   # rotated case
Since the switch to nested lifetimes, Mondrian-type numbers differ from logs made before it (same
distribution, different random draws: a fit at a larger lifetime consumes the random stream differently).
RF numbers still reproduce exactly.
NOTE: scores of the CART variant are in y^2 units, widths in x units, so lifetime grids differ.
"""
import sys
import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from adaptive_mondrian import AdaptiveMondrianForestRegressor, _predict_tree

N_TR_LIST = [3200, 6400]  # training sizes; each is a separate run (the original setting was 800)
N_VAL, N_TE = 400, 1000
M_LIST = [10, 30, 50]
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DIMS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 20, 30, 50]
ROTATED = len(sys.argv) > 3 and sys.argv[3] == "rotated"
TAG = "ROTATED " if ROTATED else ""

WIDTH_LIFETIMES = (0.5, 1000, 12)       # (low, high, points) of the geometric grid at n_train = 800
CART_LIFETIMES_D10 = (0.1, 5000, 12)    # same, multiplied by D/10 in lifetimes()
SK_GRID = [(mf, msl) for mf in (1.0, 0.33, 0.1) for msl in (1, 3, 5)]
BETA = 10  # exponential feature choice of Mondrian-CART-exp

ORDER = ["Mondrian", "Mondrian-CART", "Mondrian-CART-null", "Mondrian-CART-exp", "RF"]
ADAPTIVE = ["Mondrian-CART", "Mondrian-CART-null", "Mondrian-CART-exp"]

mse = lambda a, b: float(np.mean((a - b) ** 2))


def rotation(D, seed):
    """D x 3 matrix with orthonormal columns (a random 3-dim subspace). It has its own RNG, so the
    stream that generates X and the noise is identical to the axis-aligned experiment."""
    Q, _ = np.linalg.qr(np.random.RandomState(10000 + seed).randn(D, D))
    return Q[:, :3]


def latent(X, B):
    """The 3 coordinates the target depends on: columns 0,1,2 if B is None (axis-aligned),
    otherwise the projections 0.5 + (X - 0.5) @ B (rotated)."""
    return X[:, :3] if B is None else 0.5 + (X - 0.5) @ B


def make(rng, n, D, B=None):
    X = rng.rand(n, D)
    u = latent(X, B)
    y = np.sin(4 * u[:, 0]) + 2 * (u[:, 1] - 0.5) ** 2 + u[:, 2] + 0.1 * rng.randn(n)
    return X, y


def lifetime_grid(lo, hi, n_pts, N):
    """The original geometric grid (lo .. hi, n_pts points, tuned for n_train = 800), extended upwards
    with the same spacing until it reaches hi * N/800. The original points are kept exactly."""
    step = (hi / lo) ** (1.0 / (n_pts - 1))
    extra = max(0, int(np.ceil(np.log(N / 800.0) / np.log(step) - 1e-9)))
    return lo * step ** np.arange(n_pts + extra)


def lifetimes(score, D, N):
    if score == "width":
        return lifetime_grid(*WIDTH_LIFETIMES, N)
    lams = lifetime_grid(*CART_LIFETIMES_D10, N) * (D / 10.0)
    if score == "cart_null":
        # one extra grid point: lifetime = inf, i.e. grow until no feature beats the noise baseline.
        # An "edge" pick of this last point means "fully grown, no lifetime needed".
        lams = np.append(lams, np.inf)
    return lams


def forest_curves(Xtr, ytr, Xva, yva, Xte, yte, score, lams, seed, beta=None):
    """Validation/test mse for every (M in M_LIST, lifetime) using nested forests: the forest is grown
    once with the largest lifetime of the grid, and every smaller lifetime is obtained by truncating
    the trees at that lifetime (see _predict_tree)."""
    Mmax = max(M_LIST)
    val = np.zeros((len(M_LIST), len(lams)))
    te = np.zeros((len(M_LIST), len(lams)))
    f = AdaptiveMondrianForestRegressor(Mmax, max(lams), score=score, random_state=seed, beta=beta).fit(Xtr, ytr)
    for i, lam in enumerate(lams):
        pv = np.array([_predict_tree(t, Xva, lam) for t in f.trees_])
        pt = np.array([_predict_tree(t, Xte, lam) for t in f.trees_])
        for k, M in enumerate(M_LIST):
            val[k, i] = mse(pv[:M].mean(0), yva)
            te[k, i] = mse(pt[:M].mean(0), yte)
    return val, te


def pick(val, te, lams):
    """Choose lifetime by validation mse, per M. -> {M: (test_mse, lifetime, edge, note)}
    edge is "low" / "high" if the chosen lifetime is the first / last grid value, else None."""
    out = {}
    for k, M in enumerate(M_LIST):
        i = int(np.argmin(val[k]))
        edge = "low" if i == 0 else ("high" if i == len(lams) - 1 else None)
        out[M] = (float(te[k, i]), float(lams[i]), edge, None)
    return out


def tune_sklearn(cls, M, Xtr, ytr, Xva, yva, Xte, yte, seed):
    """Pick (max_features, min_samples_leaf) on validation; return (test_mse, note)."""
    best = None
    for mf, msl in SK_GRID:
        m = cls(M, max_features=mf, min_samples_leaf=msl, random_state=seed).fit(Xtr, ytr)
        v = mse(m.predict(Xva), yva)
        if best is None or v < best[0]:
            best = (v, m, mf, msl)
    return mse(best[1].predict(Xte), yte), f"mf={best[2]},msl={best[3]}"


def run_seed(seed, D, N_TR):
    t0 = time.time()
    rng = np.random.RandomState(seed)
    B = rotation(D, seed) if ROTATED else None
    Xtr, ytr = make(rng, N_TR, D, B)
    Xva, yva = make(rng, N_VAL, D, B)
    Xte, yte = make(rng, N_TE, D, B)
    res = {}

    variants = [
        ("Mondrian", "width", None),
        ("Mondrian-CART", "cart", None),
        ("Mondrian-CART-null", "cart_null", None),
        ("Mondrian-CART-exp", "cart", BETA),
    ]
    for name, score, beta in variants:
        lams = lifetimes(score, D, N_TR)
        val, te = forest_curves(Xtr, ytr, Xva, yva, Xte, yte, score, lams, seed, beta)
        res[name] = pick(val, te, lams)

    # sklearn random forest, tuned on validation (one set of fits per M)
    res["RF"] = {}
    for M in M_LIST:
        t, note = tune_sklearn(RandomForestRegressor, M, Xtr, ytr, Xva, yva, Xte, yte, seed)
        res["RF"][M] = (t, None, None, note)

    print(f"  n={N_TR} d={D} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return res


def setting_text(entries):
    """Median lifetime for Mondrian-type forests, most common chosen setting otherwise."""
    if entries[0][1] is not None:
        return f"lifetime {np.median([e[1] for e in entries]):.3g}"
    notes = [e[3] for e in entries if e[3]]
    return Counter(notes).most_common(1)[0][0] if notes else "-"


def edge_text(entries):
    if entries[0][1] is None:
        return "-"
    lo = sum(e[2] == "low" for e in entries)
    hi = sum(e[2] == "high" for e in entries)
    return f"{lo} low / {hi} high"


def summarize_dim(D, N_TR, results):
    n = len(results)
    print(f"\n########## {TAG}d = {D} (3 latent coordinates), n_train={N_TR}, noise var=0.01, {n} seeds ##########")
    print("test MSE mean +- std over seeds (lower is better)")
    for M in M_LIST:
        print(f"\n=== n = {N_TR}, d = {D}, M = {M} trees ===")
        print(f"{'method':20s} {'test MSE':>18s}   {'chosen setting':>22s}   {'grid-edge picks':>16s}")
        for name in ORDER:
            entries = [r[name][M] for r in results]
            v = np.array([e[0] for e in entries])
            sd = v.std(ddof=1) if n > 1 else 0.0
            print(f"{name:20s} {v.mean():10.4f} +- {sd:6.4f}   {setting_text(entries):>22s}   {edge_text(entries):>16s}")

        def paired(a, b):
            d_ = np.array([r[a][M][0] - r[b][M][0] for r in results])
            se = d_.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return f"{d_.mean():+.4f} +- {se:.4f}"

        print("paired differences (first minus second), mean +- s.e.:")
        for a, b in [(name, other) for name in ADAPTIVE for other in ("Mondrian", "RF")] + \
                    [("Mondrian-CART-null", "Mondrian-CART"),
                     ("Mondrian-CART-exp", "Mondrian-CART"),
                     ("Mondrian-CART-exp", "Mondrian-CART-null")]:
            print(f"  {a} - {b}: {paired(a, b)}")


def summarize_all(all_results, N_TR):
    dims = sorted(all_results)
    for M in (max(M_LIST), min(M_LIST)):
        print(f"\n=== {TAG}n = {N_TR}: summary across dimensions, mean test MSE, M = {M} trees ===")
        print(f"{'method':20s}" + "".join(f"{'d=' + str(D):>10s}" for D in dims))
        for name in ORDER:
            row = [np.mean([r[name][M][0] for r in all_results[D]]) for D in dims]
            print(f"{name:20s}" + "".join(f"{v:10.4f}" for v in row))


def summarize_gap(all_results, N_TR):
    """Share of the gap between the standard Mondrian and RF closed by each adaptive variant."""
    dims = sorted(all_results)
    for M in (max(M_LIST), min(M_LIST)):
        print(f"\n=== {TAG}n = {N_TR}: share of the gap (Mondrian -> RF) closed, M = {M} trees ===")
        print("0% = same as Mondrian, 100% = same as RF, above 100% = better than RF")
        print(f"{'method':20s}" + "".join(f"{'d=' + str(D):>10s}" for D in dims))
        for name in ADAPTIVE:
            row = []
            for D in dims:
                w = np.mean([r["Mondrian"][M][0] for r in all_results[D]])
                rf = np.mean([r["RF"][M][0] for r in all_results[D]])
                m = np.mean([r[name][M][0] for r in all_results[D]])
                row.append(100.0 * (w - m) / (w - rf) if abs(w - rf) > 1e-12 else float("nan"))
            print(f"{name:20s}" + "".join(f"{v:9.0f}%" for v in row))


if __name__ == "__main__":
    for N_TR in N_TR_LIST:
        all_results = {}
        for D in DIMS:
            print(f"\n##### n = {N_TR}, d = {D}: running {N_SEEDS} seeds, M in {M_LIST}", flush=True)
            all_results[D] = Parallel(n_jobs=-1)(delayed(run_seed)(s, D, N_TR) for s in range(N_SEEDS))
            summarize_dim(D, N_TR, all_results[D])
        summarize_all(all_results, N_TR)
        summarize_gap(all_results, N_TR)
