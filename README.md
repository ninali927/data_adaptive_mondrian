# Data-Adaptive Mondrian Forest

A Mondrian forest for regression where the split-feature distribution adapts to the data.

A standard Mondrian forest picks the split feature with probability proportional to the width of the
cell's bounding box, so it ignores `y`. Here the width is replaced by a data-dependent score:

$$
s_j = \mathbb{E}_{T \sim \mathrm{Unif}[\min_j, \max_j]}\big[\mathrm{SSE}(\text{parent}) - \mathrm{SSE}(\text{left}_T) - \mathrm{SSE}(\text{right}_T)\big]
$$

which is the expected CART variance reduction of a *uniformly random* split along feature `j`. Everything
else is a Mondrian process:

- wait time ~ `Exp(rate = sum_j s_j / d)`
- feature `j` ~ `s_j / sum_k s_k`
- threshold ~ `Unif[min_j, max_j]`

The score is computed exactly in `O(d n log n)` per cell, with no minimisation over thresholds.

## Variants

| `score` / `beta` | Name in the experiments | Description |
|---|---|---|
| `score="width"` | Mondrian | Standard Mondrian forest (score = bounding-box width) |
| `score="cart"` | Mondrian-CART | Data-adaptive score `s_j`, feature chosen with `p_j ∝ s_j` |
| `score="cart_null"` | Mondrian-CART-null | `max(s_j - SSE_parent/(n-1), 0)`: the reduction above what a pure-noise feature gives in expectation |
| `score="cart", beta=10` | Mondrian-CART-exp | Same score, feature chosen with `p_j ∝ exp(beta * s_j / max_k s_k)` |

CART-null and CART-exp are two separate fixes for the same problem: when `d` is large, the many noise
features have small but nonzero scores and together take most of the probability.

## Results (summary)

Synthetic data: `X ~ Unif[0,1]^d`, `y = sin(4 u0) + 2 (u1 - 0.5)^2 + u2 + noise`, where `u0, u1, u2`
are the first three columns of `X` (axis-aligned) or three random orthonormal projections (rotated).
Test MSE, mean over 10 seeds, `n_train = 800`, 50 trees, lifetime / RF hyperparameters tuned on validation:

| Method | Axis d=10 | Axis d=50 | Rotated d=10 | Rotated d=50 |
|---|---|---|---|---|
| Mondrian-CART      | 0.0270     | 0.0588     | 0.1622     | 0.4035     |
| Mondrian-CART-null | **0.0232** | 0.0358     | 0.1477     | 0.3810     |
| Mondrian-CART-exp  | 0.0263     | **0.0310** | **0.1439** | **0.3712** |
| RF (tuned)         | 0.0259     | 0.0329     | 0.1696     | 0.3714     |

See [report_exp.tex](report_exp.tex) for details. Raw logs are in [results/](results/).

## Files

| File | Purpose |
|---|---|
| [adaptive_mondrian.py](adaptive_mondrian.py) | `AdaptiveMondrianForestRegressor` and the exact score computation (numba-accelerated if available) |
| [compare.py](compare.py) | Main benchmark: Mondrian vs adaptive variants vs tuned random forest |
| [diagnostic.py](diagnostic.py) | Fraction of splits that land on the relevant features, by depth |
| [viz_partition_2d.py](viz_partition_2d.py) | 2D pictures of the partitions (stripes / disk / xor) |
| [report_exp.tex](report_exp.tex) | Short write-up of the exponential feature-choice variant |
| [results/](results/) | Experiment logs and figures |

## Installation

Python 3.11 was used.

```bash
conda create -n adaptive_mondrian python=3.11
conda activate adaptive_mondrian
pip install -r requirements.txt
```

`numba` is optional. Without it the score is computed in pure numpy (same results, slower).

## Usage

```python
from adaptive_mondrian import AdaptiveMondrianForestRegressor

model = AdaptiveMondrianForestRegressor(n_estimators=50, lifetime=10.0, score="cart", beta=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Trees are nested in lifetime: predictions for a smaller lifetime without refitting
y_pred_short = model.predict(X_test, lifetime=1.0)
```

Note: CART scores are in units of `y^2` and widths in units of `x`, so `lifetime` has a different scale
for `score="width"` and for the CART scores.

### Running the experiments

```bash
python compare.py                          # 10 seeds, d in 10,20,30,50 (axis-aligned)
python compare.py 2                        # quick run with 2 seeds
python compare.py 10 10,20,30,50 rotated   # rotated target
python diagnostic.py                       # 5 seeds, d in 10,20,30,50
python diagnostic.py 3 20,50               # 3 seeds, d in 20,50
python viz_partition_2d.py                 # 2D partition figures -> results/
python viz_partition_2d.py --leaves 25 --n 800 --noise 0.05
```
