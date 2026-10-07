"""A2 P2: joint LR-WD optima from local bivariate quadratics, and the product law in tokens."""
from collections import defaultdict

import numpy as np

from experiments.a2.provided_sweeps import load


FIXED_LR = .003
TEST_TOKENS = 2.4576e9


def points_by_budget(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["tokens"]].append((row["learning_rate"], row["weight_decay"], row["final_val_loss"]))
    return {tokens: np.array(sorted(points)) for tokens, points in sorted(groups.items())}


def fit_quadratic(points):
    """Least-squares a+bx+cy+dx^2+exy+fy^2 with x=log(LR), y=log(WD); returns (a,b,c,d,e,f)."""
    x, y = np.log(points[:, 0]), np.log(points[:, 1])
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    return np.linalg.lstsq(design, points[:, 2], rcond=None)[0]


def quadratic_optimum(coefficients):
    """Stationary point of the fitted surface as (LR, WD, loss); NaN unless it is a minimum."""
    a, b, c, d, e, f = coefficients
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    if np.any(np.linalg.eigvalsh(hessian) <= 0):
        return np.nan, np.nan, np.nan
    x, y = np.linalg.solve(hessian, [-b, -c])
    return np.exp(x), np.exp(y), a + b * x + c * y + d * x * x + e * x * y + f * y * y


def fit_power_law(tokens, values):
    exponent, log_coefficient = np.polyfit(np.log(tokens), np.log(values), 1)
    return exponent, log_coefficient


def main():
    budgets = points_by_budget(load("P2a"))
    optima = {tokens: quadratic_optimum(fit_quadratic(points)) for tokens, points in budgets.items()}
    for tokens, (lr, wd, loss) in optima.items():
        print(f"{tokens / 1e6:7.1f}M  lr*={lr:.5f}  wd*={wd:.4f}  lr*wd*={lr * wd:.3e}  L*={loss:.4f}")

    tokens = np.array(list(optima))
    products = np.array([lr * wd for lr, wd, _ in optima.values()])
    exponent, log_coefficient = fit_power_law(tokens, products)
    predicted_product = np.exp(log_coefficient + exponent * np.log(TEST_TOKENS))
    print(f"product law: lr*wd* ∝ D^{exponent:.3f}; at {TEST_TOKENS / 1e9:g}B -> {predicted_product:.3e}"
          f" -> WD {predicted_product / FIXED_LR:.4f} at LR {FIXED_LR}")

    smallest = budgets[min(budgets)]
    lr, wd, loss = smallest[smallest[:, 2].argmin()]
    print(f"best measured pair at {min(budgets) / 1e6:g}M: LR {lr:g}, WD {wd:g} (loss {loss:.4f})")


if __name__ == "__main__":
    main()
