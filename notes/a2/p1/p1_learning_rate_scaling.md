# Problem 1: When can training on less data guide large scale runs?

## (a) Small budgets

![AdamW final validation loss vs peak LR at each token budget](p1a_lr_loss_curves.png)

Final validation loss, supplied AdamW runs (WD 0.1; lowest per budget in bold):

| Peak LR | 153.6M | 307.2M | 614.4M |
| --- | ---: | ---: | ---: |
| 0.0015 | 3.2442 | 3.0709 | 2.9473 |
| 0.003 | **3.2291** | **3.0562** | **2.9256** |
| 0.006 | 3.3056 | 3.0715 | 2.9408 |

![AdamW fitted optimal LR vs token budget](p1a_optimal_lr_scaling.png)

Proposed rule (power law through the fitted optima): η\*(D) = 0.00328 · (D / 614.4M)^0.213. The optimal LR rises with token count.

## (b) Predictions at larger budgets

Recorded before inspecting the data, from the rule in (a): 0.00380 at 1.2288B, 0.00414 at 1.8432B, 0.00440 at 2.4576B.

![AdamW final validation loss vs peak LR at the P1(b) budgets](p1b_lr_loss_curves.png)

| Peak LR | 1.2288B | 1.8432B | 2.4576B |
| --- | ---: | ---: | ---: |
| 0.0015 | 2.8508 | 2.8060 | 2.7760 |
| 0.003 | **2.8378** | **2.7967** | **2.7683** |
| 0.006 | 2.8397 | 2.8008 | 2.7775 |

![AdamW predicted vs fitted optimal LR](p1b_predicted_vs_fitted.png)

The trend does not continue. The prediction holds at 1.2288B (0.00380 vs fitted 0.00388), but the fitted optimum then falls to 0.00343 and 0.00291, so the rule overpredicts by 21% and 51%.

## (c) Two scaling rules

![Power laws fit over all six budgets and over the larger three, with 4.9152B predictions](p1c_two_rules.png)

| Fit | Rule | Predicted η\* at 4.9152B |
| --- | --- | --- |
| All six budgets | η\* ∝ D^0.099 | 0.00373 |
| Larger three (1.2288B–2.4576B) | η\* ∝ D^−0.409 | 0.00223 |

The larger-three fit should do better: the optimum stopped rising near 1.2B tokens, and the all-six fit averages rising and falling optima, so it still predicts an increase.

![Final validation loss vs peak LR at 4.9152B, with both predictions](p1c_target_lr_loss.png)

| Peak LR | Source | Final val loss | Gap vs best grid LR |
| --- | --- | ---: | ---: |
| 0.0015 | grid | 2.7170 | best grid |
| 0.00223 | larger-three | **2.7156** | −0.0014 |
| 0.003 | grid | 2.7174 | +0.0004 |
| 0.00373 | all-six | 2.7205 | +0.0035 |
| 0.006 | grid | 2.7268 | +0.0098 |

The fitted target optimum is 0.00208 (±1σ 0.00182–0.00235); the larger-three prediction is 7% above it, the all-six prediction 79% above.

- **Gains from the smaller-budget fits:** none. The larger-three gain (0.0014) is within seed noise (SD 0.0024), and the all-six fit, which leans on the smallest budgets, was 0.0035 worse.
- **Bias–variance:** the all-six fit has low variance (exponent ±1σ 0.08–0.14) but is biased, since one power law cannot follow an optimum that rises then falls. The larger-three fit has less bias but high variance (−0.71 to −0.24). Bias dominated.
- **Extrapolation:** the local fit held for one 2× step. Because the trend reversed inside the source range, trust a fit only a short distance past its data. The loss is flat near the optimum (LRs 0.0015–0.003 within 0.0018 of the best), so modest LR errors cost little here.

## (d) Hyperball versus AdamW

![Hyperball final validation loss vs peak LR at each token budget](p1d_lr_loss_curves.png)

Supplied Hyperball runs (WD 0; lowest per budget in bold):

| Peak LR | 153.6M | 307.2M | 614.4M |
| --- | ---: | ---: | ---: |
| 0.0003 | 4.6079 | 4.1946 | 3.8001 |
| 0.001 | 3.9460 | 3.5495 | 3.2863 |
| 0.0015 | 3.7064 | 3.3939 | 3.1730 |
| 0.003 | 3.4326 | 3.2063 | 3.0364 |
| 0.006 | 3.2596 | 3.0835 | 2.9494 |
| 0.01 | 3.2113 | 3.0357 | **2.9176** |
| 0.015 | **3.2040** | **3.0315** | 2.9183 |
| 0.03 | 3.2533 | 3.0981 | 2.9962 |

![Hyperball vs AdamW fitted optimal LR vs token budget](p1d_optimal_lr_scaling.png)

| Tokens | Hyperball η\* | AdamW η\* |
| --- | --- | --- |
| 153.6M | 0.0137 | 0.00238 |
| 307.2M | 0.0129 | 0.00298 |
| 614.4M | 0.0121 | 0.00319 |
| Rule | ∝ D^−0.089 | ∝ D^0.213 |

Hyperball's optimum falls slowly (×0.94 per doubling) while AdamW's rises (×1.16). Hyperball's two doublings give nearly the same ratio (0.944, 0.937); AdamW's do not (1.25, 1.07).

Predicted optimum at 1.2288B, recorded before running the target: 0.0114.

![Hyperball final validation loss vs peak LR at 1.2288B, with the prediction](p1d_target_lr_loss.png)

| Peak LR | Final val loss | Gap vs best sampled |
| --- | ---: | ---: |
| 0.006 | 2.8441 | +0.0160 |
| 0.01 | 2.8284 | +0.0003 |
| 0.0114 (predicted) | **2.8281** | best |
| 0.015 | 2.8358 | +0.0076 |

- **Prediction vs target:** the predicted LR gave the best sampled loss; the fitted optimum is 0.0108 (±1σ 0.0092–0.0118), 5% lower, and LR 0.01 is within noise. Hyperball's best loss is 0.0097 below AdamW's best sampled loss at this budget (2.8378).
- **More regular:** yes. Per doubling the Hyperball optimum changes by 0.94, 0.94, 0.90; AdamW's changed by 1.25, 1.07, 1.22, then reversed (0.75). But AdamW's rule also held at 1.2288B, so one target cannot show whether Hyperball keeps its trend past where AdamW's broke.
- **Uncertainty:** three budgets give an exponent of −0.089 (±1σ −0.13 to −0.02), so the direction is clear and the rate is not. Fitted optima move ±4–6% under seed noise, and four points give −0.134. AdamW shows a trend can reverse just past the source range.

## Synthesis

AdamW's optimal LR rose with budget up to about 1.2B tokens (0.00238 at 153.6M to 0.00388 at 1.2288B), then fell (0.00291 at 2.4576B, 0.00208 at 4.9152B), so larger budgets preferred lower LRs. Hyperball's fell slowly and steadily over its whole range (0.0137 to 0.0108, about ×0.94 per doubling). Because the AdamW trend reversed, extrapolating from small budgets failed: the rule from (a) overpredicted the 2.4576B optimum by 51%, and the all-six fit predicted an LR 79% above the 4.9152B optimum, while the fit on the three largest budgets came within 7%. Far-out predictions need source points near the target budget, and a power law should be trusted only a short distance past its data. Hyperball looked more regular, but it passed only one 2× extrapolation, as AdamW did before its trend broke, so three small budgets cannot rule out a later bend.
