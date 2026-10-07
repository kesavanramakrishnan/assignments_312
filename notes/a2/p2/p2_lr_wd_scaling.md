# Problem 2: How should learning rate and weight decay scale together?

## (a) Let LR and WD vary together

![Local quadratic fits of final validation loss in (log LR, log WD) at each token budget, with fitted optima](p2a_lr_wd_contours.png)

![Fitted optimal LR and WD vs token budget, with Problem 1's optimal LR at WD .1](p2a_optima_vs_tokens.png)

| Tokens | Fitted η\* (±1σ) | Fitted λ\* (±1σ) | Problem 1 η\* (WD 0.1) |
| --- | --- | --- | --- |
| 153.6M | 0.00186 (0.00183–0.00189) | 0.847 (0.825–0.867) | 0.00238 |
| 307.2M | 0.00160 (0.00134–0.00178) | 0.519 (0.452–0.660) | 0.00298 |
| 614.4M | 0.00247 (0.00226–0.00258) | 0.258 (0.229–0.332) | 0.00319 |
| 1.2288B | 0.00270 (0.00203–0.00349) | 0.162 (0.119–0.216) | 0.00388 |
| Single power law | ∝ D^0.22 | ∝ D^−0.82 | |

**Optimal LR and WD vs scale.** Optimal WD decreases steadily with budget, from 0.85 at 153.6M to 0.16 at 1.2288B (×0.50–0.63 per doubling). Optimal LR increases overall, from 0.00186 to 0.00270, but not monotonically: it dips to 0.00160 at 307.2M.

**LR trend vs Problem 1.** Both rise at a similar overall rate (D^0.22 here, D^0.21 in Problem 1(a)). The joint optimum is 22–46% lower at every budget, where its WD is also above Problem 1's 0.1.

**Single power law?** WD, yes: it falls by a similar factor each doubling and stays close to D^−0.82. LR, no: the dip at 307.2M breaks the trend, and the 1.2288B optimum is poorly constrained (±1σ 0.0020–0.0035).

![Best measured validation loss from joint LR-WD tuning vs Problem 1, and their difference](p2a_best_loss_vs_p1.png)

| Tokens | Best joint (LR, WD) | Best joint loss | Best Problem 1 loss (LR 0.003, WD 0.1) | Benefit of tuning WD |
| --- | --- | ---: | ---: | ---: |
| 153.6M | (0.0015, 1.6) | 3.1844 | 3.2291 | 0.0447 |
| 307.2M | (0.0015, 0.8) | 3.0282 | 3.0562 | 0.0281 |
| 614.4M | (0.003, 0.2) | 2.9178 | 2.9256 | 0.0078 |
| 1.2288B | (0.003, 0.2) | 2.8361 | 2.8378 | 0.0017 |

**Benefit of tuning WD vs budget.** The benefit shrinks as the budget grows, from 0.0447 at 153.6M to 0.0017 at 1.2288B, which is within seed noise (SD 0.0024). Tuning WD converges to the point where it no longer matters, because the optimal WD falls toward the 0.1 that Problem 1 used (0.16 at 1.2288B).

## (b) From coupling to a joint scaling law

![Quadratic fits with the line ηλ = η\*λ\* and the fitted valley direction through each optimum](p2b_contours_product_line.png)

Fitted loss rise for a 2× move away from the optimum. The valley slope is d log λ / d log η along the fitted surface's flattest direction; the line ηλ = const has slope −1.

| Tokens | Valley slope | Along ηλ = η\*λ\* | LR only | WD only |
| --- | ---: | ---: | ---: | ---: |
| 153.6M | −2.33 | +0.0376 | +0.0468 | +0.0195 |
| 307.2M | −1.86 | +0.0098 | +0.0148 | +0.0069 |
| 614.4M | −2.81 | +0.0136 | +0.0196 | +0.0055 |
| 1.2288B | −1.11 | +0.0020 | +0.0086 | +0.0072 |

![Optimal LR, optimal WD, and their product vs token budget, each with a single power-law fit](p2b_lr_wd_product_scaling.png)

Single power-law fits over the four budgets (R² in log-log space; ±1σ from seed-noise refits):

| Variable | Exponent | R² (±1σ) | Change per doubling |
| --- | ---: | --- | --- |
| η\* | +0.22 | 0.678 (0.25–0.78) | ×0.86, ×1.54, ×1.09 |
| λ\* | −0.82 | 0.994 (0.82–0.99) | ×0.61, ×0.50, ×0.63 |
| η\*λ\* | −0.59 | 0.967 (0.90–0.98) | ×0.53, ×0.77, ×0.69 |

## (c) Test the joint rule

## Synthesis
