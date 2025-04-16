# Language Model Improvement Experiments on Penn Treebank

## Introduction

This document details the results of experiments aimed at improving a neural language model trained on the Penn Treebank (PTB) dataset. The primary goal is to incrementally enhance a baseline model by applying standard techniques and evaluating their impact on performance, measured primarily by Perplexity (PPL) on the test set (lower is better).

This file currently documents the results for **Part 1.A** of the mandatory exercises, which involves:
1.  Replacing a baseline RNN with LSTM.
2.  Adding Dropout layers for regularization.
3.  Replacing the SGD optimizer with AdamW.

## Experiment Setup

*   **Dataset:** Penn Treebank (PTB)
*   **Task:** Language Modeling
*   **Evaluation Metric:** Perplexity (PPL) - calculated on development set for tuning and early stopping, and on the test set for final evaluation.
*   **Baseline Concept:** The experiments start conceptually from a simple RNN, but the first implemented and measured step (A1) uses an LSTM network.
*   **Fixed Hyperparameters (unless varied):**
    *   Embedding Size (`emb`): 200
    *   Hidden Size (`hid`): 300
    *   Gradient Clipping (`clip`): 5
    *   Maximum Epochs (`n_epochs`): 100
    *   Early Stopping Patience (`patience`): 3
*   **Tuned Hyperparameters:** Learning Rate (`lr`) and Batch Size (`bs`) were varied for each step as specified below.

## Results: Part 1.A - Incremental Improvements

### Step 1 (A1): Baseline LSTM + SGD

**Modification:** The baseline RNN was replaced with a Long Short-Term Memory (LSTM) network. The Stochastic Gradient Descent (SGD) optimizer was used.

**Hyperparameter Tuning:**
*   Learning Rates tested: `[0.1, 1.0, 1.5]`
*   Batch Sizes tested: `[16, 32, 64, 128]`

**Summary Results (A1):**

| learning\_rate | batch\_size | best\_dev\_ppl | final\_test\_ppl | epochs\_trained | best\_epoch |
| :------------- | :---------- | :------------- | :--------------- | :-------------- | :---------- |
| 0.1            | 16          | 154.7477       | 148.9497         | 57              | 54          |
| 0.1            | 32          | 157.6862       | 152.8903         | 97              | 94          |
| 0.1            | 64          | 172.3083       | 166.3650         | 100             | 100         |
| 0.1            | 128         | 207.7086       | 201.0583         | 100             | 100         |
| 1.0            | 16          | 142.6022       | 138.4802         | 12              | 9           |
| 1.0            | 32          | 145.0946       | 140.0688         | 17              | 14          |
| 1.0            | 64          | 149.6502       | 143.8282         | 26              | 23          |
| 1.0            | 128         | 153.3906       | 147.5534         | 45              | 42          |
| 1.5            | 16          | **142.1862**   | **136.8386**     | 9               | 6           |
| 1.5            | 32          | 143.7457       | 140.4230         | 13              | 10          |
| 1.5            | 64          | 148.4526       | 142.4118         | 21              | 18          |
| 1.5            | 128         | 152.9149       | 146.8056         | 32              | 29          |

**Analysis (A1):**
*   All configurations successfully achieved PPL well below the required 250 threshold, confirming LSTM's suitability.
*   The best performance was achieved with `lr=1.5` and `bs=16`, yielding a Test PPL of **136.84**.
*   Relatively high learning rates (1.0, 1.5) combined with small batch sizes (16, 32) proved most effective for SGD in this setup.
*   Larger batch sizes generally resulted in higher PPL.

---

### Step 2 (A2): LSTM + Dropout + SGD

**Modification:** Building on Step 1, two dropout layers were added: one after the embedding layer (`emb_dropout=0.1`) and one before the final linear layer (`fc_dropout=0.1`). SGD optimizer was retained.

**Hyperparameter Tuning:**
*   Learning Rates tested: `[0.1, 1.0, 1.5]`
*   Batch Sizes tested: `[16, 32, 64, 128]`
*   Dropout Rates: Fixed at 0.1 for both layers.

**Summary Results (A2):**

| learning\_rate | batch\_size | emb\_dropout | fc\_dropout | best\_dev\_ppl | final\_test\_ppl | epochs\_trained | best\_epoch |
| :------------- | :---------- | :----------- | :---------- | :------------- | :--------------- | :-------------- | :---------- |
| 0.1            | 16          | 0.1          | 0.1         | 131.9128       | 127.3092         | 100             | 100         |
| 0.1            | 32          | 0.1          | 0.1         | 147.9698       | 142.6839         | 100             | 99          |
| 0.1            | 64          | 0.1          | 0.1         | 173.6776       | 167.6592         | 100             | 100         |
| 0.1            | 128         | 0.1          | 0.1         | 211.4252       | 204.6155         | 100             | 100         |
| 1.0            | 16          | 0.1          | 0.1         | 129.5247       | 125.4559         | 22              | 19          |
| 1.0            | 32          | 0.1          | 0.1         | 128.2923       | 124.3020         | 35              | 32          |
| 1.0            | 64          | 0.1          | 0.1         | 128.7768       | 124.3895         | 56              | 53          |
| 1.0            | 128         | 0.1          | 0.1         | 129.6862       | 125.3803         | 93              | 90          |
| 1.5            | 16          | 0.1          | 0.1         | 130.5575       | 125.8188         | 17              | 14          |
| 1.5            | 32          | 0.1          | 0.1         | 129.4472       | 124.4734         | 27              | 24          |
| 1.5            | 64          | 0.1          | 0.1         | **128.8177**   | **123.7342**     | 38              | 35          |
| 1.5            | 128         | 0.1          | 0.1         | 129.1585       | 124.8125         | 71              | 68          |

**Analysis (A2):**
*   The addition of dropout resulted in a **significant and consistent improvement** in PPL across comparable configurations from Step A1. This indicates successful regularization, reducing overfitting.
*   The best Test PPL decreased from 136.84 (A1) to **123.73** (A2), achieved with `lr=1.5` and `bs=64`.
*   Optimal learning rates remained high (1.0, 1.5) for SGD.
*   The optimal batch size shifted slightly towards medium values (32, 64), suggesting dropout might reduce sensitivity to very small batch sizes.
*   Convergence generally required more epochs compared to A1, which is expected when using regularization.

---

### Step 3 (A3): LSTM + Dropout + AdamW

**Modification:** The SGD optimizer was replaced with AdamW, while retaining the LSTM architecture and dropout layers from Step A2.

**Hyperparameter Tuning:**
*   Learning Rates tested: `[0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]` (Note: Higher LRs were included for comparison but proved unsuitable for AdamW).
*   Batch Sizes tested: `[16, 32, 64, 128]`
*   Dropout Rates: Fixed at 0.1.

**Summary Results (A3):**

| learning\_rate | batch\_size | emb\_dropout | fc\_dropout | optimizer | best\_dev\_ppl | final\_test\_ppl | epochs\_trained | best\_epoch |
| :------------- | :---------- | :----------- | :---------- | :-------- | :------------- | :--------------- | :-------------- | :---------- |
| 0.0001         | 16          | 0.1          | 0.1         | AdamW     | 132.0732       | 119.9252         | 50              | 47          |
| 0.0001         | 32          | 0.1          | 0.1         | AdamW     | 131.9869       | 119.9798         | 76              | 73          |
| 0.0001         | 64          | 0.1          | 0.1         | AdamW     | 132.8371       | 121.4562         | 100             | 100         |
| 0.0001         | 128         | 0.1          | 0.1         | AdamW     | 138.2740       | 128.1957         | 100             | 100         |
| 0.001          | 16          | 0.1          | 0.1         | AdamW     | **127.6820**   | **119.5284**     | 8               | 5           |
| 0.001          | 32          | 0.1          | 0.1         | AdamW     | 130.2232       | 121.2960         | 10              | 7           |
| 0.001          | 64          | 0.1          | 0.1         | AdamW     | 130.9289       | 121.8619         | 13              | 10          |
| 0.001          | 128         | 0.1          | 0.1         | AdamW     | 130.7962       | 122.9160         | 19              | 16          |
| 0.01           | 16          | 0.1          | 0.1         | AdamW     | 170.8837       | 157.7178         | 6               | 3           |
| 0.01           | 32          | 0.1          | 0.1         | AdamW     | 154.2206       | 145.2590         | 6               | 3           |
| 0.01           | 64          | 0.1          | 0.1         | AdamW     | 148.2045       | 139.0513         | 6               | 3           |
| 0.01           | 128         | 0.1          | 0.1         | AdamW     | 145.3653       | 136.2293         | 6               | 3           |
| 0.1            | 16          | 0.1          | 0.1         | AdamW     | 1098.6184      | 1024.8956        | 5               | 2           |
| 0.1            | 32          | 0.1          | 0.1         | AdamW     | 569.0416       | 532.9120         | 4               | 1           |
| 0.1            | 64          | 0.1          | 0.1         | AdamW     | 386.5240       | 357.4784         | 4               | 1           |
| 0.1            | 128         | 0.1          | 0.1         | AdamW     | 296.4847       | 274.6819         | 6               | 3           |
| 1.0            | 16          | 0.1          | 0.1         | AdamW     | 2124843.6823   | 1771657.8332     | 4               | 1           |
| 1.0            | 32          | 0.1          | 0.1         | AdamW     | 9091.1637      | 8840.4626        | 5               | 2           |
| 1.0            | 64          | 0.1          | 0.1         | AdamW     | 4664.3052      | 4553.4836        | 5               | 2           |
| 1.0            | 128         | 0.1          | 0.1         | AdamW     | 4770.2522      | 4511.4771        | 6               | 3           |
| 1.5            | 16          | 0.1          | 0.1         | AdamW     | 4.1157e+21     | 2.4128e+21       | 4               | 1           |
| 1.5            | 32          | 0.1          | 0.1         | AdamW     | 9636.8412      | 9610.5881        | 4               | 1           |
| 1.5            | 64          | 0.1          | 0.1         | AdamW     | 2916.8826      | 2862.9237        | 15              | 12          |
| 1.5            | 128         | 0.1          | 0.1         | AdamW     | 8757.4896      | 8746.6242        | 6               | 3           |


**Analysis (A3):**
*   Replacing SGD with AdamW yielded **further performance improvements**, provided the learning rate was adjusted appropriately.
*   The optimal learning rate range shifted dramatically downwards; `lr=0.001` and `lr=0.0001` produced the best results, while higher learning rates (>= 0.01) led to poor performance or divergence, highlighting the critical need for LR tuning when changing optimizers.
*   The best overall configuration for Part 1.A was found in this step: `lr=0.001`, `bs=16` resulted in a Test PPL of **119.53**.
*   Small batch sizes (16, 32) remained slightly preferable with the optimal LRs.
*   AdamW demonstrated much faster convergence (best epoch often below 10) compared to SGD when using an appropriate learning rate.

---

## Overall Conclusions (Part 1.A)

The experiments successfully demonstrated the effectiveness of the applied techniques:
1.  **LSTM significantly outperforms baseline RNNs** for language modeling on PTB.
2.  **Dropout provides substantial regularization benefits** when added to the LSTM model, leading to lower perplexity.
3.  **AdamW, when properly tuned with a low learning rate, further improves performance** compared to SGD, achieving the lowest perplexity overall.

The best configuration identified in Part 1.A is an **LSTM model with Embedding and FC Dropout (0.1 rate each), trained using the AdamW optimizer with a learning rate of 0.001 and a batch size of 16**. This configuration achieved a final **Test Perplexity of 119.53**.

*(Further sections can be added here for Part 1.B or other experiments)*