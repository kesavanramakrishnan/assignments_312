# Assignment 1 run index

Completed runs with a recorded final validation loss. Interrupted runs are omitted. A run reused later is listed once.

Unless a row says otherwise, the configuration is d8 (8 layers, width 512), 614.4M fresh tokens, AdamW, learning rate 0.003, 1% warmup, linear decay, batch size 64, weight decay 0.1, no dropout, and model/data seed 42.

## Problem 1(a): one-variable sweeps

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 1A-01 | learning rate 0.001 | 2.9360 | 2.9777 |
| 1A-02 | learning rate 0.000333 | 3.0909 | 3.1265 |
| 1A-03 | warmup 0.1% | 2.9507 | 2.9945 |
| 1A-04 | warmup 0.3% | 2.8995 | 2.9402 |
| 1A-05 | standard recipe | 2.8859 | 2.9259 |
| 1A-06 | learning rate 0.009 | 2.9049 | 2.9493 |
| 1A-07 | learning rate 0.027 | 2.9735 | 3.0172 |
| 1A-08 | warmup 3.0% | 2.8815 | 2.9245 |
| 1A-09 | warmup 10.0% | 2.8788 | 2.9202 |
| 1A-10 | batch 16 | 2.7977 | 2.9272 |
| 1A-11 | batch 32 | 2.8631 | 2.9185 |
| 1A-12 | batch 128 | 2.9087 | 2.9497 |
| 1A-13 | weight decay 0.01 | 2.9036 | 2.9466 |
| 1A-14 | weight decay 0.03 | 2.9051 | 2.9420 |
| 1A-15 | weight decay 0.3 | 2.8761 | 2.9200 |
| 1A-16 | weight decay 1 | 2.9160 | 2.9624 |
| 1A-17 | batch 256 | 3.0050 | 3.0136 |

## Problem 1(b): hyperparameter interactions

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 1B-01 | learning rate 0.001; batch 256 | 3.0983 | 3.1051 |
| 1B-02 | learning rate 0.009; batch 256 | 3.0967 | 3.1046 |
| 1B-03 | learning rate 0.027; batch 256 | 3.1269 | 3.1350 |
| 1B-04 | learning rate 0.001; batch 32 | 2.8895 | 2.9429 |
| 1B-05 | learning rate 0.009; batch 32 | 2.8989 | 2.9514 |
| 1B-06 | learning rate 0.027; batch 32 | 2.9739 | 3.0276 |
| 1B-07 | learning rate 0.009; warmup 0.1% | 2.9542 | 2.9922 |
| 1B-08 | learning rate 0.009; warmup 10.0% | 2.8670 | 2.9135 |
| 1B-09 | learning rate 0.027; warmup 0.1% | 2.9625 | 3.0054 |
| 1B-10 | learning rate 0.027; warmup 10.0% | 2.9122 | 2.9588 |
| 1B-11 | learning rate 0.001; weight decay 0.01 | 2.9526 | 2.9944 |
| 1B-12 | learning rate 0.001; weight decay 1 | 2.8873 | 2.9326 |
| 1B-13 | learning rate 0.009; weight decay 0.01 | 2.9413 | 2.9855 |
| 1B-14 | learning rate 0.009; weight decay 1 | 3.1170 | 3.1599 |
| 1B-15 | learning rate 0.027; weight decay 0.01 | 2.9682 | 3.0140 |
| 1B-16 | learning rate 0.027; weight decay 1 | 3.2966 | 3.3343 |
| 1B-17 | learning rate 0.001; batch 128 | 2.9835 | 3.0249 |
| 1B-18 | learning rate 0.009; batch 128 | 2.9429 | 2.9873 |

## Problem 1(c): learning-rate schedules

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 1C-01 | learning rate 0.001; cosine schedule | 2.9410 | 2.9833 |
| 1C-02 | cosine schedule | 2.8909 | 2.9328 |
| 1C-03 | learning rate 0.001; constant schedule | 3.0316 | 3.0659 |
| 1C-04 | learning rate 0.009; cosine schedule | 2.9067 | 2.9497 |
| 1C-05 | constant schedule | 3.1424 | 3.1754 |
| 1C-06 | warmup-stable-decay (20% decay) | 2.8926 | 2.9358 |
| 1C-07 | learning rate 0.001; warmup-stable-decay (20% decay) | 2.9051 | 2.9499 |
| 1C-08 | learning rate 0.009; constant schedule | 3.2627 | 3.2932 |
| 1C-09 | learning rate 0.009; warmup-stable-decay (20% decay) | 2.9470 | 2.9906 |
| 1C-10 | learning rate 0.027; constant schedule | 3.4480 | 3.4741 |
| 1C-11 | learning rate 0.027; warmup-stable-decay (20% decay) | 3.0653 | 3.1121 |
| 1C-12 | constant schedule; weight decay 0.3 | 3.2375 | 3.2753 |
| 1C-13 | warmup-stable-decay (20% decay); weight decay 0.3 | 2.9192 | 2.9644 |
| 1C-14 | constant schedule; weight decay 1 | 3.4484 | 3.4861 |
| 1C-15 | warmup-stable-decay (20% decay); weight decay 1 | 3.0113 | 3.0593 |
| 1C-16 | warmup-stable-decay (10% decay) | 2.9081 | 2.9554 |
| 1C-17 | warmup-stable-decay (40% decay) | 2.8747 | 2.9216 |
| 1C-18 | learning rate 0.009; warmup-stable-decay (20% decay); warmup 0.0%; weight decay 1 | 3.3788 | 3.4174 |
| 1C-19 | learning rate 0.009; warmup-stable-decay (20% decay); warmup 20.0%; weight decay 1 | 3.1639 | 3.2082 |
| 1C-20 | learning rate 0.009; warmup 10.0%; weight decay 1 | 3.0117 | 3.0566 |

## Problem 2(a): model-size ladders

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 2A-01 | 4 layers, width 256 | 3.2539 | 3.2910 |
| 2A-02 | 4 layers, width 256; dropout 0.2 | 3.5533 | 3.4465 |
| 2A-03 | 4 layers, width 256; learning rate 0.03 | 3.2766 | 3.3178 |
| 2A-04 | 4 layers, width 256; constant schedule | 3.3908 | 3.4256 |
| 2A-05 | 5 layers, width 320; constant schedule | 3.2950 | 3.3315 |
| 2A-06 | 5 layers, width 320; dropout 0.2 | 3.4096 | 3.3257 |
| 2A-07 | 5 layers, width 320 | 3.1110 | 3.1538 |
| 2A-08 | 5 layers, width 320; learning rate 0.03 | 3.1591 | 3.2011 |
| 2A-09 | 6 layers, width 384; constant schedule | 3.2053 | 3.2472 |
| 2A-10 | 6 layers, width 384 | 3.0175 | 3.0562 |
| 2A-11 | 6 layers, width 384; learning rate 0.03 | 3.0660 | 3.1096 |
| 2A-12 | 6 layers, width 384; dropout 0.2 | 3.2908 | 3.2163 |
| 2A-13 | 7 layers, width 448 | 2.9417 | 2.9854 |
| 2A-14 | 7 layers, width 448; constant schedule | 3.1751 | 3.2132 |
| 2A-15 | 7 layers, width 448; dropout 0.2 | 3.1910 | 3.1294 |
| 2A-16 | 7 layers, width 448; learning rate 0.03 | 3.0166 | 3.0571 |
| 2A-17 | learning rate 0.03 | 2.9916 | 3.0307 |
| 2A-18 | dropout 0.2 | 3.1125 | 3.0630 |
| 2A-19 | 9 layers, width 576 | 2.8436 | 2.8837 |
| 2A-20 | 9 layers, width 576; constant schedule | 3.0930 | 3.1280 |
| 2A-21 | 9 layers, width 576; dropout 0.2 | 3.0466 | 3.0033 |
| 2A-22 | 9 layers, width 576; learning rate 0.03 | 2.9574 | 3.0009 |

## Problem 2(b): alternate scaling axes

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 2B-01 | 75,000 sequences (76.8M tokens processed) | 3.4948 | 3.5344 |
| 2B-02 | 4 layers, width 256; cosine schedule | 3.2522 | 3.2939 |
| 2B-03 | 1,200,000 sequences (1.23B tokens processed) | 2.8325 | 2.8371 |
| 2B-04 | 2,400,000 sequences (2.46B tokens processed) | 2.7565 | 2.7682 |
| 2B-05 | 150,000 sequences (154M tokens processed) | 3.1957 | 3.2307 |
| 2B-06 | 300,000 sequences (307M tokens processed) | 3.0878 | 3.0584 |
| 2B-07 | 6 layers, width 384; cosine schedule | 3.0183 | 3.0613 |
| 2B-08 | 5 layers, width 320; cosine schedule | 3.1193 | 3.1574 |
| 2B-09 | 9 layers, width 576; cosine schedule | 2.8537 | 2.8917 |
| 2B-10 | 7 layers, width 448; cosine schedule | 2.9474 | 2.9904 |

## Problem 2(c): scaling-law failures

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 2C-01 | 75,000 sequences × 2 epochs (154M tokens processed) | 3.1562 | 3.2431 |
| 2C-02 | 75,000 sequences × 16 epochs (1.23B tokens processed) | 2.5468 | 3.1607 |
| 2C-03 | 75,000 sequences × 4 epochs (307M tokens processed) | 2.9157 | 3.1121 |
| 2C-04 | 75,000 sequences × 8 epochs (614M tokens processed) | 2.7004 | 3.0896 |
| 2C-05 | 75,000 sequences × 32 epochs (2.46B tokens processed) | 2.4084 | 3.2504 |
| 2C-06 | 4 layers, width 256; 75,000 sequences (76.8M tokens processed) | 3.7230 | 3.7638 |
| 2C-07 | 5 layers, width 320; 75,000 sequences (76.8M tokens processed) | 3.5964 | 3.6277 |
| 2C-08 | 6 layers, width 384; 75,000 sequences (76.8M tokens processed) | 3.5460 | 3.5756 |
| 2C-09 | 7 layers, width 448; 75,000 sequences (76.8M tokens processed) | 3.5471 | 3.5871 |
| 2C-10 | 9 layers, width 576; 75,000 sequences (76.8M tokens processed) | 3.5451 | 3.5838 |
| 2C-11 | 8 layers, width 256 | 3.1054 | 3.1473 |
| 2C-12 | 8 layers, width 320 | 3.0288 | 3.0719 |
| 2C-13 | 8 layers, width 384 | 2.9737 | 3.0176 |
| 2C-14 | 8 layers, width 448 | 2.9218 | 2.9663 |
| 2C-15 | 8 layers, width 576 | 2.8600 | 2.8993 |
| 2C-16 | 8 layers, width 768 | 2.8048 | 2.8455 |
| 2C-17 | 4 layers, width 512 | 3.0120 | 3.0556 |
| 2C-18 | 5 layers, width 512 | 2.9752 | 3.0157 |
| 2C-19 | 6 layers, width 512 | 2.9344 | 2.9783 |
| 2C-20 | 7 layers, width 512 | 2.9084 | 2.9515 |
| 2C-21 | 9 layers, width 512 | 2.8662 | 2.9065 |
| 2C-22 | 12 layers, width 512 | 2.8323 | 2.8733 |

## Problem 3(a): joint variation

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 3A-01 | model/data seed 1 | 2.8978 | 2.9285 |
| 3A-02 | model/data seed 2 | 2.8833 | 2.9303 |
| 3A-03 | model/data seed 3; A100 | 2.8799 | 2.9328 |

## Problem 3: deterministic references

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 3REF-01 | deterministic execution | 2.8858 | 2.9271 |
| 3REF-02 | deterministic execution | 2.8858 | 2.9271 |

## Problem 3(b): isolated variation

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 3B-01 | model seed 1, data seed 42; deterministic execution | 2.8819 | 2.9244 |
| 3B-02 | model seed 0, data seed 42; deterministic execution | 2.8873 | 2.9298 |
| 3B-03 | model seed 2, data seed 42; deterministic execution | 2.8902 | 2.9301 |
| 3B-04 | model seed 42, data seed 0; deterministic execution | 2.9330 | 2.9309 |
| 3B-05 | model seed 42, data seed 1; deterministic execution | 2.8939 | 2.9272 |
| 3B-06 | model seed 42, data seed 2; deterministic execution | 2.8762 | 2.9286 |

## Problem 3(c): variability under interventions

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 3C-01 | dropout 0.2; model/data seed 0 | 3.1608 | 3.0639 |
| 3C-02 | dropout 0.2; model/data seed 1 | 3.1148 | 3.0553 |
| 3C-03 | dropout 0.2; model/data seed 2 | 3.0998 | 3.0609 |
| 3C-04 | learning rate 0.027; model/data seed 0 | 3.0059 | 3.0009 |
| 3C-05 | learning rate 0.027; model/data seed 2 | 2.9489 | 3.0019 |
| 3C-06 | learning rate 0.027; model/data seed 1 | 2.9721 | 3.0114 |
| 3C-07 | 12 layers, width 768; model/data seed 0 | 2.7949 | 2.7884 |
| 3C-08 | 12 layers, width 768; model/data seed 1 | 2.7634 | 2.7967 |
| 3C-09 | 12 layers, width 768; model/data seed 2 | 2.7398 | 2.7880 |
| 3C-10 | 2,400,000 sequences (2.46B tokens); model/data seed 0 | 2.7355 | 2.7683 |
| 3C-11 | 2,400,000 sequences (2.46B tokens); model/data seed 1 | 2.7587 | 2.7700 |
| 3C-12 | 2,400,000 sequences (2.46B tokens); model/data seed 2 | 2.7035 | 2.7731 |

## Problem 3(c): seeds 4–9

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 3S-01 | model/data seed 4 | 2.9759 | 2.9293 |
| 3S-02 | batch 16; model/data seed 4 | 2.9721 | 2.9340 |
| 3S-03 | learning rate 0.009; model/data seed 4 | 2.9842 | 2.9447 |
| 3S-04 | batch 16; model/data seed 5 | 3.1215 | 2.9322 |
| 3S-05 | model/data seed 5 | 2.9386 | 2.9323 |
| 3S-06 | learning rate 0.009; model/data seed 5 | 2.9537 | 2.9470 |
| 3S-07 | model/data seed 6 | 2.9094 | 2.9326 |
| 3S-08 | batch 16; model/data seed 6 | 2.7982 | 2.9324 |
| 3S-09 | learning rate 0.009; model/data seed 6 | 2.9323 | 2.9513 |
| 3S-10 | model/data seed 7 | 2.8833 | 2.9357 |
| 3S-11 | batch 16; model/data seed 7 | 2.8650 | 2.9417 |
| 3S-12 | learning rate 0.009; model/data seed 7 | 2.9044 | 2.9579 |
| 3S-13 | model/data seed 8 | 2.8737 | 2.9348 |
| 3S-14 | learning rate 0.009; model/data seed 8 | 2.8797 | 2.9437 |
| 3S-15 | batch 16; model/data seed 8 | 2.7214 | 2.9356 |
| 3S-16 | model/data seed 9 | 2.8871 | 2.9334 |
| 3S-17 | batch 16; model/data seed 9 | 2.9430 | 2.9318 |

## Problem 4(a): one-token perturbation

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 4A-01 | deterministic execution; one training token changed | 2.8858 | 2.9254 |
| 4A-02 | deterministic execution | 2.8859 | 2.9277 |

## Problem 4(b): weight perturbation during training

Noise scale is a fraction of each parameter tensor's standard deviation. Training then continues to the end.

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 4B-01 | deterministic; perturb at step 100, scale 0.001 | 2.8863 | 2.9258 |
| 4B-02 | deterministic; perturb at step 100, scale 0.01 | 2.8836 | 2.9261 |
| 4B-03 | deterministic; perturb at step 100, scale 0.1 | 2.8845 | 2.9259 |
| 4B-04 | deterministic; perturb at step 2000, scale 0.001 | 2.8847 | 2.9262 |
| 4B-05 | deterministic; perturb at step 2000, scale 0.01 | 2.8841 | 2.9264 |
| 4B-06 | deterministic; perturb at step 2000, scale 0.1 | 2.8850 | 2.9265 |
| 4B-07 | deterministic; perturb at step 5000, scale 0.001 | 2.8840 | 2.9262 |
| 4B-08 | deterministic; perturb at step 5000, scale 0.01 | 2.8838 | 2.9258 |
| 4B-09 | deterministic; perturb at step 5000, scale 0.1 | 2.8847 | 2.9268 |
| 4B-10 | deterministic; perturb at step 8000, scale 0.001 | 2.8840 | 2.9262 |
| 4B-11 | deterministic; perturb at step 8000, scale 0.01 | 2.8839 | 2.9262 |
| 4B-12 | deterministic; perturb at step 8000, scale 0.1 | 2.8875 | 2.9301 |

## Problem 5: diagnostic loss curves

| ID | Changes from the standard recipe | Final train loss | Final validation loss |
| --- | --- | ---: | ---: |
| 5-01 | learning rate 0.0003 | 3.1116 | 3.1473 |

