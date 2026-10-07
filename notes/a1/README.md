# Assignment 1: Basics — empirical machine learning

Kesavan Ramakrishnan

This submission studies the course d8 language model through controlled hyperparameter, scaling, stochasticity, and diagnostic experiments. Unless stated otherwise, the standard run uses 8 layers of width 512, 614.4M fresh training tokens, AdamW, learning rate 0.003, 1% warmup, linear decay, batch size 64, weight decay 0.1, and no dropout.

## Reports

- [Problem 1: Hyperparameter sweeps](p1/p1_hyperparameters.md)
- [Problem 2: Reliability of scaling laws](p2_scaling_law_reliability.md)
- [Problem 3: Measuring variation](p3_measuring_variation.md)
- [Problem 4: Amplification of randomness](p4_amplification.md)
- [Problem 5: Loss-curve augury](p5_loss_curve_augury.md)
- [Problem 6: Activations and gradient norms](p6_activation_gradient_norms.md)
- [Completed-run index](run_index.md)

## Main conclusions

Learning rate and batch size have the largest direct effects on d8 performance, while warmup and weight decay matter mainly through interactions. Model-size scaling is highly predictable when the data budget is sufficient, but repeated or insufficient data breaks the trend. Ordinary seed-to-seed validation variation is a few thousandths of a nat, and tiny deterministic perturbations can produce differences of that same order. Loss-curve shape separates schedule and learning-rate problems from batch noise, while internal RMS statistics show residual growth with depth, large mid-training activations, and small late gradients.
