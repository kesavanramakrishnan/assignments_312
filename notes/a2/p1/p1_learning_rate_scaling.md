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

| Fit | Rule | Predicted $\eta^*$ at 4.9152B |
| --- | --- | --- |
| All six budgets | $\eta^* \propto D^{0.099}$ | 0.00373 |
| Larger three (1.2288B–2.4576B) | $\eta^* \propto D^{-0.409}$ | 0.00223 |

**Which prediction should give the lower loss?** The larger-three prediction. The optimal LR stopped rising after about 1.2B tokens, and the larger-three fit follows the trend closest to the target. The all-six fit averages the rising small-budget optima with the falling larger ones, so it still predicts an increase.

![AdamW final validation loss vs peak LR at 4.9152B, with both predictions](p1c_target_lr_loss.png)

| Peak LR | Source | Final validation loss | Gap vs best grid LR |
| --- | --- | ---: | ---: |
| 0.0015 | target grid | 2.7170 | best grid LR |
| 0.00223 | larger-three prediction | **2.7156** | −0.0014 |
| 0.003 | target grid | 2.7174 | +0.0004 |
| 0.00373 | all-six prediction | 2.7205 | +0.0035 |
| 0.006 | target grid | 2.7268 | +0.0098 |

The fitted target optimum is 0.00208 ($\pm1\sigma$ 0.00182–0.00235). The larger-three prediction is 7% above it and the all-six prediction is 79% above it.

**Gains from the smaller-budget fits.** The larger-three prediction beat the best grid LR by 0.0014, which is within seed noise (SD 0.0024). The all-six prediction, which relies on the smallest budgets, was 0.0035 worse. Including the small budgets did not help.

**Bias–variance.** The all-six fit has low variance (exponent $\pm1\sigma$ 0.08–0.14) but is biased, because one power law cannot follow an optimum that rises and then falls. The larger-three fit has less bias but high variance (exponent $\pm1\sigma$ −0.71 to −0.24). Here bias dominated, so the noisier local fit predicted better.

**Extrapolation.** The local fit held for a 2× extrapolation, from 2.4576B to 4.9152B. Because the trend reversed within the source budgets, a fit should only be trusted a short distance past the budgets it was fit on. Near the optimum the loss is flat: 0.0015–0.003 are within 0.0018 of the best run, so a modest LR error costs little at this budget.

## (d) Hyperball versus AdamW

![Hyperball final validation loss vs peak LR at each token budget](p1d_lr_loss_curves.png)

![Hyperball vs AdamW fitted optimal LR vs token budget](p1d_optimal_lr_scaling.png)

Predicted optimal LR at 1.2288B from the Hyperball power law, recorded before running the target: 0.0114.

## (e) Explore schedule variants (optional)

## Synthesis

