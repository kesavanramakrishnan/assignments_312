# Problem 1: When can training on less data guide large scale runs?

## (a) Learn from small training budgets

![AdamW final validation loss vs peak LR at each token budget](p1a_lr_loss_curves.png)

![AdamW fitted optimal LR vs token budget](p1a_optimal_lr_scaling.png)

Fitting a power law to the fitted optima gives the proposed scaling rule

$$
\eta^*(D) = 0.00328\left(\frac{D}{614.4\text{M}}\right)^{0.21}
$$

From what we see, the optimal LR rises with token count.

## (b) Predict before increasing the budget

Predicted optimal LRs from the power law in (a), recorded before inspecting the P1(b) measurements:

| Tokens | Predicted $\eta^*$ |
| --- | --- |
| 1.2288B | 0.00380 |
| 1.8432B | 0.00414 |
| 2.4576B | 0.00440 |

## (c) Compare and test two scaling rules

## (d) Hyperball versus AdamW

## (e) Explore schedule variants (optional)

## Synthesis

