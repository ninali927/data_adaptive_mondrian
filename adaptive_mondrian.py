"""
Data-adaptive Mondrian forest for regression.

Same stochastic process as a Mondrian forest (exponential clock, random feature
choice, uniform random threshold), except that the per-feature "box width" is
replaced by a data-dependent score:

    s_j = E_{T ~ Unif[min_j, max_j]} [ SSE(parent) - SSE(left_T) - SSE(right_T) ]

i.e. the expected CART variance reduction of a *uniformly random* split along j.
(min_j, max_j) is the bounding box of the training points in the cell, exactly as
in the Mondrian-kernel code this builds on, so the integration domain of the score
is the same interval the threshold is sampled from.

Per cell:
    wait time ~ Exp(rate = sum_j s_j / d)      (rate_by_d=True, as in the meeting)
    feature j ~ s_j / sum_k s_k
    threshold ~ Unif[min_j, max_j]
Stop splitting if the cell has < 2 points, or all scores are 0, or the clock
exceeds `lifetime`.

score="width" gives the ordinary Mondrian (score = bounding-box width) for comparison.

score="cart_null" is a VARIANT that goes beyond the algorithm as originally described: the same score
minus the noise baseline b = SSE(parent)/(n-1), clipped at 0,
    s_j' = max(s_j - b, 0).
For a feature that is unrelated to y, the expected reduction of a split is exactly b (permutation
argument: with k = 2 groups, E[between-group SS] = SST/(n-1); both sides of the split are non-empty because
the threshold is drawn inside the data's bounding box). So s_j' is the expected reduction *above chance*:
pure-noise features get weight close to zero, and a cell stops splitting when no feature beats chance.

beta (exponential feature choice) is a VARIANT of the feature-choice step only. With beta=None (default)
the feature is drawn with p_j = s_j / sum_k s_k as above. With a number beta >= 0,
    p_j ∝ exp(beta * s_j / max_k s_k)   for features with s_j > 0   (p_j = 0 if s_j = 0).
The exponential widens the gap between high- and low-score features, so the many low-score (noise)
features together get much less probability than under proportional sampling. Features with equal
scores still get equal probability. Scores are divided by the cell's largest score first, because raw
scores shrink with the number of points in the cell: exp(s_j) would be almost uniform in deep cells.
The exponential clock is unchanged (rate still uses the raw scores), so lifetime grids stay valid.

NOTE: scores are in units of y^2 (raw SSE, not normalised), widths in units of x.
So `lifetime` has a different scale in the two modes and depends on the scale of y.
"""
import numpy as np
from sklearn.utils.validation import check_random_state

try:
    from numba import njit
except ImportError:  # numba missing: fall back to the pure numpy loop (same results, slower)
    njit = None

SCORES = ("width", "cart", "cart_null")


def _raw_scores_numpy(X, yc, S):
    """Per-feature loop of cart_scores (before null correction and clipping), pure numpy."""
    n, d = X.shape
    s = np.zeros(d)
    k = np.arange(1, n)
    for j in range(d):
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        width = xs[-1] - xs[0]
        if width <= 0:
            continue
        Sk = np.cumsum(yc[order])[:-1]
        gain = Sk**2 / k + (S - Sk) ** 2 / (n - k) - S**2 / n
        s[j] = np.diff(xs) @ gain / width
    return s


if njit is not None:
    @njit(cache=True)
    def _raw_scores_numba(X, yc, S):
        """Same as _raw_scores_numpy, compiled. Mergesort is stable, so ties are ordered as in numpy."""
        n, d = X.shape
        s = np.zeros(d)
        k = np.arange(1, n).astype(np.float64)
        for j in range(d):
            col = np.ascontiguousarray(X[:, j])
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            width = xs[-1] - xs[0]
            if width <= 0:
                continue
            Sk = np.cumsum(yc[order])[:-1]
            gain = Sk**2 / k + (S - Sk) ** 2 / (n - k) - S**2 / n
            s[j] = np.dot(np.diff(xs), gain) / width
        return s
    _raw_scores = _raw_scores_numba
else:
    _raw_scores = _raw_scores_numpy


def cart_scores(X, y, null_correct=False):
    """
    Exact s_j for every feature j of the cell containing (X, y).
    If null_correct=True, return max(s_j - SSE_parent/(n-1), 0) instead (see module docstring).

    SSE(parent) - SSE(left) - SSE(right) is piecewise constant in T: it only
    changes when T crosses a data point. With points sorted along j and a split
    after the k-th point (k left, n-k right),
        reduction_k = S_k^2/k + (S-S_k)^2/(n-k) - S^2/n,   S_k = sum of first k y's.
    s_j = sum_k (x_(k+1) - x_(k)) * reduction_k / (x_(n) - x_(1)).
    Cost O(d n log n). No minimisation over T. The loop over features runs in numba if it is installed.
    """
    n, d = X.shape
    if n < 2:
        return np.zeros(d)
    yc = y - y.mean()  # centring is only for numerical stability
    S = yc.sum()
    s = _raw_scores(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(yc, dtype=np.float64), S)
    if null_correct:
        # noise baseline SSE_parent/(n-1); constant-width features (s=0) end up clipped to 0
        s = s - (yc @ yc) / (n - 1)
    return np.maximum(s, 0.0)


def feature_probs(sc, beta=None):
    """Feature-choice probabilities from the scores of a cell (sum of sc must be > 0).
    beta=None -> proportional to the scores (original version); otherwise exponential (module docstring)."""
    if beta is None:
        return sc / sc.sum()
    # subtracting beta (the largest exponent) only avoids overflow, the probabilities are unchanged
    w = np.where(sc > 0, np.exp(beta * (sc / sc.max() - 1.0)), 0.0)
    return w / w.sum()


def _grow_tree(X, y, lifetime, score, rate_by_d, rng, beta=None):
    n, d = X.shape
    feature, thr, left, right, value, time = [], [], [], [], [], []

    def new_node():
        feature.append(-1); thr.append(0.0); left.append(-1); right.append(-1); value.append(0.0)
        time.append(np.inf)  # split time of the node; inf for leaves
        return len(feature) - 1

    root = new_node()
    stack = [(root, np.arange(n), 0.0)]
    while stack:
        node, idx, t = stack.pop()
        Xc, yc = X[idx], y[idx]
        value[node] = yc.mean()
        if len(idx) < 2:
            continue
        if score in ("cart", "cart_null"):
            sc = cart_scores(Xc, yc, null_correct=(score == "cart_null"))
        else:
            sc = Xc.max(0) - Xc.min(0)
        total = sc.sum()
        if not total > 0:
            continue
        rate = total / d if rate_by_d else total
        t_cut = t + rng.exponential(1.0 / rate)
        if t_cut >= lifetime:
            continue
        j = rng.choice(d, p=feature_probs(sc, beta))
        lo, hi = Xc[:, j].min(), Xc[:, j].max()
        loc = lo + (hi - lo) * rng.rand()
        mask = Xc[:, j] <= loc
        if mask.all() or not mask.any():  # measure-zero guard
            continue
        l, r = new_node(), new_node()
        feature[node], thr[node], left[node], right[node], time[node] = j, loc, l, r, t_cut
        stack.append((l, idx[mask], t_cut))
        stack.append((r, idx[~mask], t_cut))
    return dict(feature=np.array(feature), thr=np.array(thr), left=np.array(left),
                right=np.array(right), value=np.array(value), time=np.array(time))


def _predict_tree(tree, X, lifetime=np.inf):
    """Prediction of the tree truncated at `lifetime`: a node whose split time is >= lifetime is used as
    a leaf (its value is the mean of y in its cell). A tree grown with a larger lifetime and truncated at
    lifetime has the same distribution as a tree grown with that lifetime (nested lifetimes), so one fit
    at the largest lifetime of a grid gives predictions for every lifetime of the grid."""
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


class AdaptiveMondrianForestRegressor:
    """
    score : "cart"      -> data-adaptive variant (expected CART reduction)
            "cart_null" -> same, minus the noise baseline SSE_parent/(n-1), clipped at 0 (variant)
            "width"     -> standard Mondrian (bounding-box width)
    rate_by_d : divide the exponential rate by d (as described in the meeting).
                Applied identically to both modes. (The old Mondrian-kernel code
                used the plain sum of widths, i.e. rate_by_d=False; this only
                rescales lifetime by d.)
    beta : None -> feature chosen proportionally to its score (original version)
           float -> exponential feature choice p_j ∝ exp(beta * s_j / max_k s_k) (variant, see module docstring)
    """

    def __init__(self, n_estimators=10, lifetime=1.0, score="cart", rate_by_d=True, random_state=42,
                 beta=None):
        self.n_estimators = n_estimators
        self.lifetime = lifetime
        self.score = score
        self.rate_by_d = rate_by_d
        self.random_state = random_state
        self.beta = beta

    def fit(self, X, y):
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {self.score!r}")
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.trees_ = [_grow_tree(X, y, self.lifetime, self.score, self.rate_by_d, rng, self.beta)
                       for _ in range(self.n_estimators)]
        return self

    def predict(self, X, lifetime=np.inf):
        """Forest prediction, with every tree truncated at `lifetime` (default: no truncation)."""
        X = np.asarray(X, float)
        return np.mean([_predict_tree(t, X, lifetime) for t in self.trees_], axis=0)

    def n_leaves(self):
        return float(np.mean([(t["feature"] < 0).sum() for t in self.trees_]))