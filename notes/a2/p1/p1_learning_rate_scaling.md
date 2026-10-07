# Problem 1: When can training on less data guide large scale runs?

## (a) Learn from small training budgets

![AdamW final validation loss vs peak LR at each token budget](p1a_lr_loss_curves.png)

![AdamW fitted optimal LR vs token budget](p1a_optimal_lr_scaling.png)

Fitting a power law to the fitted optima gives the proposed scaling rule

$$\eta^*(D) = 0.00328\left(\frac{D}{614.4\text{M}}\right)^{0.21}$$

The optimal LR rises with token count.

## (b) Predict before increasing the budget

## (c) Compare and test two scaling rules

## (d) Hyperball versus AdamW

## (e) Explore schedule variants (optional)

## Synthesis
