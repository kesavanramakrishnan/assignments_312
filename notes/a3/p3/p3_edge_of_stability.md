# Problem 3: The edge of stability

Thresholds: SGD ηλ_max → 2; AdamW (β₁ = 0.9, frozen preconditioner) ηλ_max^pre → 2(1+β₁)/(1−β₁) = 38.

## Predictions (written before looking at the data)

- (a)i: λ_max over training for N = 32, η = 0.01:
- (a)ii: λ_max after halving η at 750, and after doubling:
- (b): does minibatch AdamW train at the edge:
- (c): effect of LR, batch size, β₁, β₂, weight decay, clipping, warmup on η·λ_max^pre:

## (a) Full-batch replication

## (b) Minibatch AdamW

## (c) Hyperparameter exploration

## Synthesis
