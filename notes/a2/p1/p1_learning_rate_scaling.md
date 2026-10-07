# Problem 1: When can training on less data guide large scale runs?

## (a) Learn from small training budgets

![AdamW final validation loss vs peak LR at each token budget](p1a_lr_loss_curves.png)

![AdamW fitted optimal LR vs token budget](p1a_optimal_lr_scaling.png)

Fitting a power law to the fitted optima gives the proposed scaling rule

$$
\eta^*(D) = 0.00328\left(\frac{D}{614.4\text{M}}\right)^{0.213}
$$

From what we see, the optimal LR rises with token count.

## (b) Predict before increasing the budget

Predicted optimal LRs from the power law in (a), recorded before inspecting the P1(b) measurements:

| Tokens | Predicted $\eta^*$ |
| --- | --- |
| 1.2288B | 0.00380 |
| 1.8432B | 0.00414 |
| 2.4576B | 0.00440 |

![AdamW final validation loss vs peak LR at the P1(b) budgets](p1b_lr_loss_curves.png)

![AdamW predicted vs fitted optimal LR](p1b_predicted_vs_fitted.png)

The trend from the small budgets does not continue. The prediction holds at 1.2288B (predicted 0.00380, fitted 0.00388) but not after. The fitted optimum falls to 0.00343 at 1.8432B and 0.00291 at 2.4576B, so the power law overpredicts by 21% and 51%. The optimal LR stops rising after about 1.2B tokens and then decreases.

## (c) Compare and test two scaling rules

![AdamW power laws fit over all six budgets and over the larger three, with 4.9152B predictions](p1c_two_rules.png)

## (d) Hyperball versus AdamW

![Hyperball final validation loss vs peak LR at each token budget](p1d_lr_loss_curves.png)

![Hyperball vs AdamW fitted optimal LR vs token budget](p1d_optimal_lr_scaling.png)

Predicted optimal LR at 1.2288B from the Hyperball power law, recorded before running the target: 0.0114.

## (e) Explore schedule variants (optional)

## Synthesis

