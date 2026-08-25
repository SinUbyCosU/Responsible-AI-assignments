# Responsible AI Assignment — Running Report

This document is updated incrementally as each experiment is completed.
Results tables should be filled in after running the corresponding script.

---

## Q1 — Color-Biased MNIST: Do Models Learn Color Shortcuts?

**Script:** `color_biased_mnist.py`

### Setup

MNIST digits (0–9) are converted to RGB by coloring the foreground pixels:
- **Digits 0–4** → RED foreground
- **Digits 5–9** → BLUE foreground

A `bias_strength = 0.9` parameter means 90% of training images follow this rule;
the remaining 10% get the opposite color. The background stays black.

**Four models are trained:**

| Model | Training data |
|---|---|
| CNN (color) | 3-channel color-biased MNIST |
| ViT (color) | 3-channel color-biased MNIST |
| CNN (grayscale) | 1-channel plain MNIST (no color) |
| ViT (grayscale) | 1-channel plain MNIST (no color) |

**Three test conditions:**

| Condition | Description |
|---|---|
| Same bias (0.9) | Test set follows same 90/10 color rule as training (in-distribution) |
| Random color | Foreground color is fully randomized, no class correlation |
| Reversed color | Color rule inverted: 0–4 now BLUE, 5–9 now RED |

### Results Table

> Fill in after running `python color_biased_mnist.py`

| Model | Same Bias (0.9) | Random Color | Reversed Color |
|---|---|---|---|
| CNN (color) | 98.9% | 97.8% | 97.0% |
| ViT (color) | 97.3% | 93.3% | 93.2% |
| CNN (grayscale) | 99.1% | 99.1% | 99.1% |
| ViT (grayscale) | 97.6% | 97.6% | 97.6% |

### Shape vs Color Diagnostic

> Fill in after running

| Model | Correct Color | Reversed Color | Drop | Verdict |
|---|---|---|---|---|
| CNN (color) | 98.9% | 97.0% | 1.9% | Relies mostly on digit shape -> robust |
| ViT (color) | 97.3% | 93.2% | 4.1% | Relies mostly on digit shape -> robust |

### Key Findings & Interpretation

**What to look for:**
- **Large accuracy drop (IID → Reversed)** for colored models = model learned color as a shortcut,
  not the actual digit shape.
- **Grayscale models show consistent accuracy** across all 3 columns because color is invisible to them.
  This is the "clean" baseline — it tells us what accuracy is achievable from shape alone.
- **Comparing CNN vs ViT** on colored data: if ViT drops more sharply on reversed color, it may be
  more prone to texture/color shortcuts (compared to CNN which has stronger spatial inductive bias).
- The **gap between IID accuracy and random/reversed accuracy** directly measures how much the model
  relied on the spurious color correlation rather than the digit shape.

**Interpretation template (fill in after running):**
```
CNN (color):  IID=98.9% | Random=97.8% | Reversed=97.0%
  Drop (IID->Reversed): 1.9% -> robust

ViT (color):  IID=97.3% | Random=93.3% | Reversed=93.2%
  Drop (IID->Reversed): 4.1% -> robust

Grayscale models: consistent ~97-99% across all conditions (shape-only baseline).
```

---

## Q2 — Does the Background Matter? Spurious Background Correlations in Cats vs Dogs

**Script:** `q2_background_bias.py`
**Grad-CAM outputs:** `./q2_gradcam/`

### Setup

A binary classification task (cats=0, dogs=1) is built from CIFAR-10 cats (class 3)
and dogs (class 5). Each 32×32 animal image is resized to 40×40 and centered on a
64×64 solid-color canvas, giving a **12-pixel solid-color border** on every side.
This border accounts for ~61% of the total image area — plenty of signal for a model
to exploit without ever looking at the animal.

**Training bias (`bias_strength = 0.9`):**
- 90% of cats → GREEN background
- 90% of dogs → BLUE background
- 10% of each class → opposite color (to make the rule imperfect, as in real spurious correlations)

**Three test sets (all from CIFAR-10 test split):**

| Test Set | Background assignment |
|---|---|
| IID | Same 90/10 rule as training (green=cat, blue=dog) |
| Balanced | Background randomly assigned 50/50, class-independent |
| Counterfactual | Rule fully reversed: 90% cats=BLUE, 90% dogs=GREEN |

**Two models trained on the biased training set:**

| Model | Description |
|---|---|
| CNN | 3-block conv net trained from scratch on 64×64 images |
| ResNet18 | Pretrained on ImageNet, fine-tuned on biased data |

### Results Table

> Fill in after running `python q2_background_bias.py`

| Model | Test Set | Overall Acc | green-cat | blue-cat | green-dog | blue-dog |
|---|---|---|---|---|---|---|
| CNN | IID | 91.1% | 99.0% | 12.4% | 19.5% | 99.3% |
| CNN | Balanced | 58.9% | 99.6% | 13.6% | 23.0% | 99.2% |
| CNN | Counterfactual | 28.1% | 100.0% | 16.5% | 24.6% | 99.0% |
| ResNet18 | IID | 91.6% | 97.8% | 24.8% | 42.5% | 98.0% |
| ResNet18 | Balanced | 66.1% | 98.2% | 27.1% | 41.1% | 98.2% |
| ResNet18 | Counterfactual | 39.8% | 96.7% | 26.4% | 40.6% | 100.0% |

### Confusion Matrices

> Fill in after running (format: rows=true class, cols=predicted class)

**CNN — IID:**
```
           Pred:cat  Pred:dog
True:cat    899      101
True:dog     76      924
```

**CNN — Balanced:**
```
           Pred:cat  Pred:dog
True:cat    565      435
True:dog    388      612
```

**CNN — Counterfactual:**
```
           Pred:cat  Pred:dog
True:cat    241      759
True:dog    678      322
```

**ResNet18 — IID:**
```
           Pred:cat  Pred:dog
True:cat    901       99
True:dog     68      932
```

**ResNet18 — Balanced:**
```
           Pred:cat  Pred:dog
True:cat    626      374
True:dog    303      697
```

**ResNet18 — Counterfactual:**
```
           Pred:cat  Pred:dog
True:cat    328      672
True:dog    533      467
```

### Grad-CAM Visualizations

Grad-CAM heatmaps are saved in `./q2_gradcam/` with filename format:
`{ModelName}_{TestSetName}.png`

Each image shows: **[Original] [Grad-CAM heatmap] [Overlay]** for both
correctly-classified and misclassified examples.

**What to look for:**
- Heatmap lights up **border region** → model uses background color as its decision cue.
- Heatmap lights up **animal body/face** → model has learned actual visual features.
- Misclassified examples with border-focused heatmaps are the clearest evidence of
  shortcut learning.

### Key Findings & Interpretation

**What to look for:**
- **IID >> Counterfactual accuracy** = spurious correlation was learned.
  A 30%+ drop strongly suggests the model is using background as the primary cue.
- **Per-group breakdown**: in IID test,
  - `green-cat` and `blue-dog` (correct-correlation groups) should be high
  - `blue-cat` and `green-dog` (wrong-correlation groups) should be low
  - A large gap between these pairs confirms color-shortcut reliance.
- **CNN vs ResNet18**: ResNet18 was pretrained on ImageNet and already "knows" what
  cats and dogs look like. If it still drops heavily on counterfactual, that shows
  fine-tuning on biased data can overwrite robust pretrained features.
- **Balanced test set accuracy** tells us how well each model performs when the
  shortcut is simply unavailable (without being actively misleading).

**Interpretation template (fill in after running):**
```
CNN:
  IID=91.1% | Balanced=58.9% | Counterfactual=28.1%
  Drop: 63.0% -> shortcut learner
  Per-group gap on IID (green-cat vs blue-cat): 86.6% -> large gap
  Grad-CAM: heatmap focuses on border (as seen in saved images)

ResNet18:
  IID=91.6% | Balanced=66.1% | Counterfactual=39.8%
  Drop: 51.8% -> shortcut learner
  Per-group gap on IID (green-cat vs blue-cat): 73.0% -> large gap
  Grad-CAM: heatmap focuses on border
```

---

*More questions will be added here as experiments are completed.*


---

## Q3 — The Long Tail Problem: Does Class Imbalance Cause Tail Class Collapse?

**Script:** `q3_long_tail.py`

### Setup

CIFAR-10 training set subsampled to 7 classes with a long-tailed distribution:

| Class | CIFAR-10 label | Training samples | Proportion |
|---|---|---|---|
| airplane | 0 | 5000 | 52.2% |
| automobile | 1 | 2500 | 26.1% |
| bird | 2 | 1250 | 13.0% |
| cat | 3 | 500 | 5.2% |
| deer | 4 | 250 | 2.6% |
| dog | 5 | 50 | 0.5% |
| frog | 6 | 25 | 0.3% |

**Imbalance ratio: 5000 / 25 = 200:1** (head vs. tail)

> Note: Assignment Table 1 specifies [10000, 5000, ..., 50]. CIFAR-10 has at most 5000 per class, so all counts are halved. The 200:1 imbalance ratio is preserved.

**Test set:** 1000 per class × 7 = 7000 samples, perfectly balanced.

**Models:** Same CNN and TinyViT as previous questions (adapted for 7-class, 32×32 input).

### Interventions (all use identical architecture, optimizer, LR, epochs)

| Run | What changes |
|---|---|
| Baseline (CE) | Standard cross-entropy, no imbalance handling |
| Weighted CE | Per-class weights `= total / (K × n_k)` passed to CrossEntropyLoss |
| Oversampling | WeightedRandomSampler: rare classes sampled more per epoch (same CE loss) |
| Focal Loss | `(1 - p_t)^γ` modulating factor with γ=2 down-weights easy head-class examples |

### Results Table — CNN

> Fill in after running `python q3_long_tail.py`

| Class | N_train | Baseline | Weighted CE | Oversampling | Focal Loss |
|---|---|---|---|---|---|
| airplane | 5000 | 90.2% | 83.5% | 89.4% | 91.9% |
| automobile | 2500 | 92.5% | 86.6% | 88.3% | 94.5% |
| bird | 1250 | 77.2% | 46.6% | 79.1% | 71.5% |
| cat | 500 | 65.9% | 72.9% | 50.2% | 46.8% |
| deer | 250 | 41.4% | 68.1% | 34.5% | 46.7% |
| dog | 50 | 6.1% | 5.2% | 4.2% | 7.6% |
| frog | 25 | 3.6% | 9.4% | 2.4% | 7.3% |
| **Balanced Acc** | — | 53.8% | 53.2% | 49.7% | 52.3% |
| **Worst Class** | — | 3.6% | 5.2% | 2.4% | 7.3% |
| **Overall Acc** | — | 53.8% | 53.2% | 49.7% | 52.3% |

### Results Table — ViT

> Fill in after running

| Class | N_train | Baseline | Weighted CE | Oversampling | Focal Loss |
|---|---|---|---|---|---|
| airplane | 5000 | 89.9% | 73.0% | 71.4% | 87.8% |
| automobile | 2500 | 83.5% | 74.5% | 93.8% | 86.6% |
| bird | 1250 | 63.2% | 24.8% | 46.7% | 47.0% |
| cat | 500 | 46.0% | 43.7% | 29.8% | 43.8% |
| deer | 250 | 21.7% | 42.5% | 41.1% | 44.6% |
| dog | 50 | 0.6% | 11.5% | 9.7% | 3.0% |
| frog | 25 | 0.1% | 33.9% | 2.1% | 2.1% |
| **Balanced Acc** | — | 43.6% | 43.4% | 42.1% | 45.0% |
| **Worst Class** | — | 0.1% | 11.5% | 2.1% | 2.1% |
| **Overall Acc** | — | 43.6% | 43.4% | 42.1% | 45.0% |

### Combined Summary

> Fill in after running

| Run | Model | Balanced Acc | Worst Class | Overall Acc |
|---|---|---|---|---|
| Baseline (CE) | CNN | 53.8% | 3.6% | 53.8% |
| Baseline (CE) | ViT | 43.6% | 0.1% | 43.6% |
| Weighted CE | CNN | 53.2% | 5.2% | 53.2% |
| Weighted CE | ViT | 43.4% | 11.5% | 43.4% |
| Oversampling | CNN | 49.7% | 2.4% | 49.7% |
| Oversampling | ViT | 42.1% | 2.1% | 42.1% |
| Focal Loss | CNN | 52.3% | 7.3% | 52.3% |
| Focal Loss | ViT | 45.0% | 2.1% | 45.0% |

### Key Findings & Interpretation

**What to look for:**

- **Tail class collapse under baseline**: With 25 training examples for the rarest class, the baseline model often learns to predict 0% accuracy on that class entirely — it's more efficient for CE loss to always predict a head class.
- **Overall accuracy is misleading**: The baseline can score high overall because head classes dominate the test set counts, but balanced accuracy and worst-class accuracy expose the real problem.
- **Weighted CE vs Oversampling**: Both address imbalance but differently:
  - Weighted CE: each sample still appears once, but rare samples contribute more to the loss.
  - Oversampling: rare samples are literally seen multiple times per epoch.
- **Focal Loss**: Addresses imbalance indirectly — by down-weighting easy (correctly-classified head class) examples, it forces the model to focus on hard examples, which in a long-tailed setting are often the tail classes.
- **Balanced accuracy gain**: The main metric to cite. A jump from baseline balanced accuracy of X% to Y% after intervention shows how much tail-class performance improved.

**Interpretation template (fill in after running):**
```
Baseline:
  CNN worst-class accuracy: 3.6%  (baseline tail collapse)
  ViT worst-class accuracy: 0.1%

Best intervention for CNN: Focal Loss (+3.7% worst-class improvement)
Best intervention for ViT: Weighted CE (+11.4% worst-class improvement)

Test set distribution: The test set used is perfectly balanced (1000 per class), which is why Overall Acc and Balanced Acc are identical across all runs. The head-to-tail gap in baseline CNN is 90.2% (airplane) - 3.6% (frog) = 86.6%.

After best intervention (Weighted CE for ViT): worst-class jumps from 0.1% to 11.5%.
  -> Interventions reduce but do not fully eliminate tail collapse.
```
