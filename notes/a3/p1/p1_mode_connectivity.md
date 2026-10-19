# Problem 1: Mode connectivity

Midpoint barrier B(θ₁, θ₂) = L((θ₁+θ₂)/2) − (L(θ₁)+L(θ₂))/2; weight averaging works when B ≤ 0.01.
Setup: d8, 614M-token schedule, constant LR after warmup, AdamW. Prefix of x steps, two branches (batch-order seeds 1, 2) of 500 more steps; x ∈ {0, 50, 100, 250, 500, 1000}, y ∈ {25, 50, 100, 250, 500}.

## Predictions (written before looking at the data)

- B(x, y) as a function of x and y:
- Where the ensemble beats the endpoints and by how much:

## (a) Endpoint and ensemble losses

## (b) Weight averaging and the barrier

## (c) Interpolation curves

## (d) What controls whether runs share a basin

## Synthesis
