<div align="center">

# Quantitative Comparative Analysis of Classification Methods in Pattern Recognition
### HOG-Based Evaluation of Diverse Machine Learning Classifiers Across Structural, Real-World, and Texture Datasets

<br/>

[![Türkçe Dokümantasyon](https://img.shields.io/badge/Dokümantasyon-Türkçe-red?style=for-the-badge&logo=readme)](README_TR.md)

[🇹🇷 **Türkçe dokümanı okumak için buraya tıklayınız**](README_TR.md)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Domain](https://img.shields.io/badge/Domain-Pattern%20Recognition%20%7C%20Computer%20Vision-success?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Author:** Ersin Sandal  
*Academic Project & Research Study — Pattern Recognition Coursework*

</div>

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Experimental Framework & Methodology](#-experimental-framework--methodology)
  - [Pipeline Overview](#pipeline-overview)
  - [Controlled Variable: Fixed HOG Feature Extraction](#controlled-variable-fixed-hog-feature-extraction)
  - [Independent Variable: Machine Learning Classifiers](#independent-variable-machine-learning-classifiers)
  - [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Datasets](#-benchmark-datasets)
- [Quantitative Experimental Results](#-quantitative-experimental-results)
- [Comprehensive Discussion & Deep-Dive Findings](#-comprehensive-discussion--deep-dive-findings)
  - [1. Fashion-MNIST: The Regime of Linearly Separable Manifolds](#1-fashion-mnist-the-regime-of-linearly-separable-manifolds)
  - [2. CIFAR-10: Complex Visual Context and Non-Linear Distributions](#2-cifar-10-complex-visual-context-and-non-linear-distributions)
  - [3. KTH-TIPS: The Curse of Dimensionality in Texture Classification](#3-kth-tips-the-curse-of-dimensionality-in-texture-classification)
  - [4. Computational Complexity: Training Latency vs. Inference Speed](#4-computational-complexity-training-latency-vs-inference-speed)
- [Visualizations & Confusion Matrices](#-visualizations--confusion-matrices)
- [Synthesis: Validation of the 'No Free Lunch' Theorem](#-synthesis-validation-of-the-no-free-lunch-theorem)
- [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
- [Installation & Reproducibility Guide](#-installation--reproducibility-guide)
- [Presentation Slide Deck](#-presentation-slide-deck)
- [Citation & Attribution](#-citation--attribution)

---

## 🔬 Executive Summary

In machine vision and pattern recognition, end-to-end classification performance is governed by two coupled stages: **feature representation** and **decision surface learning**. Frequently, empirical benchmarks conflate feature richness with classifier capability. 

This research project presents an experimental, quantitative comparative analysis of four distinct classification paradigms:
1. **Support Vector Machines (Linear SVM)**
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest (RF)**
4. **Multi-Layer Perceptron (MLP)**

To rigorously isolate the algorithmic behavior, inductive biases, and generalization capacities of each classifier, the feature extraction pipeline is held strictly constant across all trials using **Histogram of Oriented Gradients (HOG)**. The evaluation spans three systematically divergent image domains:
- **Fashion-MNIST** (Canonical shape / silhouette features, grayscale, low complexity)
- **CIFAR-10** (Real-world object recognition, complex photographic distributions, medium complexity)
- **KTH-TIPS** (High-dimensional texture classification, $d=34,596$, high complexity)

### Key Empirical Findings:
- **No Universally Superior Classifier:** The empirical evidence strongly corroborates the *No Free Lunch Theorem*; classifier efficacy is fundamentally tied to feature space geometry and dataset characteristics.
- **Equivalence Under Linear Separability:** On well-structured, clean silhouette data (Fashion-MNIST), all four classifiers converge to nearly identical classification performance (~85.5% – 86.2% Accuracy, Macro F1 ~0.858 – 0.865).
- **Model Capacity Matters on Complex Scenes:** On unconstrained photographic data (CIFAR-10), **MLP** achieves the strongest performance (47.80% Accuracy), outperforming linear and tree-based baselines due to its capacity for non-linear representation.
- **The Curse of Dimensionality & Metric Collapse:** On high-resolution texture data (KTH-TIPS, $d=34,596$ features with $N=648$ samples), **Linear SVM** overwhelmingly dominates with **64.81% accuracy** (70.89% precision), while distance-based **KNN completely collapses to 23.46%** due to Euclidean distance concentration in ultra-high dimensions.

---

## 🎯 Motivation & Problem Statement

In applied pattern recognition, a persistent question is: *“Which classifier should be deployed for an arbitrary visual recognition task?”* Practitioners frequently switch classification models arbitrarily without understanding the interaction between feature dimensionality, sample size, and classifier inductive bias.

### The Problem:
- Different classifiers produce radically different decision boundaries on identical feature representations.
- Relying on a single metric (e.g., raw Top-1 accuracy) masks critical trade-offs between precision, recall, and computational latency.
- Suboptimal classifier selection can lead to performance degradation or catastrophic failures in production environments.

### Research Objective:
To eliminate feature representation variance as a confounding factor by **fixing HOG features** with identical hyper-parameters, thereby providing an unadulterated quantitative evaluation of classifier behavior across different data manifolds.

```
+--------------------------------------------------------------------------------------------------+
|                                    EXPERIMENTAL CONTROL PARADIGM                                 |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   [ Input Image ] ---> [ FIXED Feature Extractor: HOG ] ---> [ VARIABLE Classifier ] ---> Metric|
|                                                                                                  |
|   Fixed Parameters:                                          Algorithms:                         |
|   * orientations = 9                                         1. Linear SVM                       |
|   * pixels_per_cell = (4, 4)                                 2. K-Nearest Neighbors (k=5)        |
|   * cells_per_block = (2, 2)                                 3. Random Forest (100 trees)        |
|   * block_norm = 'L2-Hys'                                    4. Multi-Layer Perceptron (128 units|
+--------------------------------------------------------------------------------------------------+
```

---

## 🛠 Experimental Framework & Methodology

### Pipeline Overview

The execution pipeline standardizes image ingestion, channel adaptation, local contrast normalization, spatial gradient accumulation, model fitting, and multi-metric statistical reporting.

```mermaid
flowchart TD
    A["Raw Input Data<br/>(Fashion-MNIST / CIFAR-10 / KTH-TIPS)"] --> B["Image Preprocessing<br/>- Grayscale Conversion (Luminance weighting)<br/>- Intensity Normalization [0.0, 1.0]<br/>- Dimension Resizing (KTH-TIPS: 128x128)"]
    B --> C["Feature Extraction (STRICTLY FIXED)<br/>Histogram of Oriented Gradients (HOG)<br/>- Orientations: 9 bins [0°-180°]<br/>- Cell Size: 4x4 pixels<br/>- Block Size: 2x2 cells<br/>- Normalization: L2-Hys"]
    C --> D{"Classifier Arena"}
    D --> E1["Support Vector Machine<br/>(Linear Kernel)"]
    D --> E2["K-Nearest Neighbors<br/>(k = 5, Euclidean Distance)"]
    D --> E3["Random Forest<br/>(100 Estimators, Bagging)"]
    D --> E4["Multi-Layer Perceptron<br/>(128 Hidden Units, ReLU, Adam)"]
    E1 --> F["Comprehensive Evaluation Suite<br/>- Accuracy, Precision (Macro), Recall (Macro), Macro F1<br/>- Wall-Clock Train Time & Test Latency<br/>- Full Confusion Matrices & Performance Charts"]
    E2 --> F
    E3 --> F
    E4 --> F
```

---

### Controlled Variable: Fixed HOG Feature Extraction

Histogram of Oriented Gradients ([Dalal & Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)) captures local edge and gradient structure while maintaining invariance to global illumination shifts.

#### Algorithmic Parameters:
$$\text{Orientations} = 9 \quad (\text{unsigned angles in } [0^\circ, 180^\circ])$$
$$\text{Pixels per cell} = (4, 4)$$
$$\text{Cells per block} = (2, 2)$$
$$\text{Block Normalization} = \text{L2-Hys (L2-norm with clipping at 0.2 and re-normalization)}$$

#### Extracted Feature Dimensionalities:
- **Fashion-MNIST** ($28 \times 28$): $\left(\frac{28}{4} - 2 + 1\right) \times \left(\frac{28}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 6 \times 6 \times 36 = \mathbf{1,296} \text{ dimensions}$
- **CIFAR-10** ($32 \times 32$): $\left(\frac{32}{4} - 2 + 1\right) \times \left(\frac{32}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 7 \times 7 \times 36 = \mathbf{1,764} \text{ dimensions}$
- **KTH-TIPS** ($128 \times 128$): $\left(\frac{128}{4} - 2 + 1\right) \times \left(\frac{128}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 31 \times 31 \times 36 = \mathbf{34,596} \text{ dimensions}$

---

### Independent Variable: Machine Learning Classifiers

Four distinct machine learning paradigms were configured to reflect varying model capacities and inductive biases:

| Classifier | Model Paradigm | Hyperparameters | Inductive Bias / Key Characteristics |
| :--- | :--- | :--- | :--- |
| **SVM** | Maximum Margin Linear Separator | `kernel='linear'`, $C=1.0$ | Seeks the hyperplane maximizing the geometric margin between classes. Highly robust in high-dimensional feature spaces ($d \gg n$). |
| **KNN** | Instance-Based Memory Learner | `n_neighbors=5`, metric=`minkowski` ($p=2$) | Non-parametric lazy learner; assumes metric smoothness in Euclidean space. Zero training cost; computationally demanding during test. |
| **Random Forest** | Bagged Decision Tree Ensemble | `n_estimators=100`, criterion=`gini` | Aggregates 100 decorrelated CART trees with random feature subsampling. Resistant to overfitting; handles non-linear boundaries. |
| **MLP** | Feedforward Artificial Neural Network | `hidden_layer_sizes=(128,)`, `max_iter=300`, solver=`adam`, activation=`relu` | Parametric connectionist model learning composite non-linear transformations via backpropagation. |

---

### Evaluation Metrics

To eliminate class frequency bias, macro-averaged metrics are calculated across all classes $C$:

$$\text{Accuracy} = \frac{\sum_{c=1}^C TP_c}{N_{\text{total}}}$$

$$\text{Precision}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$$

$$\text{Macro } F_1 = 2 \cdot \frac{\text{Precision}_{\text{macro}} \cdot \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}$$

Additionally, **Training Wall-Clock Time** and **Inference Wall-Clock Time** are measured in seconds using high-resolution timers.

---

## 📊 Benchmark Datasets

Three benchmark datasets were selected to cover distinct structural, spatial, and dimensional properties:

```
+---------------------------------------------------------------------------------------------+
| Dataset       | Input Resolution | Channels | Train Samples | Test Samples | Classes | Nature  |
+---------------+------------------+----------+---------------+--------------+---------+---------+
| Fashion-MNIST | 28 x 28          | 1 (Gray) | 10,000*       | 2,000*       | 10      | Shape   |
| CIFAR-10      | 32 x 32          | 3 -> 1   | 10,000*       | 2,000*       | 10      | Natural |
| KTH-TIPS      | 128 x 128        | 1 (Gray) | 648 (80%)     | 162 (20%)    | 10      | Texture |
+---------------------------------------------------------------------------------------------+
*Subsampled uniformly to 10k train / 2k test to maintain experimental parity and tractable runtime.
```

1. **Fashion-MNIST:** 10 fashion product categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Sharp silhouette outlines with minimal background clutter.
2. **CIFAR-10:** 10 natural object categories (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck). High intra-class variance, varying viewpoints, lighting, and complex background noise. Converted to grayscale via standard luminance weighting:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
3. **KTH-TIPS (Textures under varying Illumination, Pose and Scale):** 10 material classes (Aluminium foil, Brown bread, Corduroy, Cotton, Cracker, Linen, Orange peel, Sandpaper, Sponge, Styrofoam). Extreme feature dimensionality ($d = 34,596$) relative to sample size ($N_{\text{train}} = 648$).

---

## 📈 Quantitative Experimental Results

Below is the complete, official benchmark matrix generated from `results/metrics/classifier_summary_results.csv`:

| Dataset | Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | Train Time (s) | Test Time (s) | Feature Dimensions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fashion-MNIST** | **SVM (Linear)** | 0.8580 | 0.8608 | 0.8599 | 0.8600 | 12.12 s | 4.04 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **KNN (k=5)** | 0.8555 | 0.8616 | 0.8569 | 0.8581 | **0.02 s** | 2.73 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **Random Forest** | 0.8575 | 0.8607 | 0.8591 | 0.8597 | 31.58 s | 0.08 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **MLP (128)** | **0.8620** | **0.8663** | **0.8636** | **0.8647** | 28.20 s | **0.01 s** | $(10000, 1296) \to (2000, 1296)$ |
| | | | | | | | | |
| **CIFAR-10** | **SVM (Linear)** | 0.4375 | 0.4428 | 0.4357 | 0.4369 | 71.10 s | 12.78 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **KNN (k=5)** | 0.3990 | 0.4960 | 0.3913 | 0.3805 | **0.03 s** | 1.08 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **Random Forest** | 0.4195 | 0.4124 | 0.4159 | 0.4113 | 57.91 s | 0.11 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **MLP (128)** | **0.4780** | **0.4750** | **0.4748** | **0.4740** | 66.38 s | **0.01 s** | $(10000, 1764) \to (2000, 1764)$ |
| | | | | | | | | |
| **KTH-TIPS** | **SVM (Linear)** | **0.6481** | **0.7089** | **0.6493** | **0.6576** | 60.75 s | 8.44 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **KNN (k=5)** | 0.2346 | 0.4126 | 0.2320 | 0.1561 | **0.04 s** | 0.46 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **Random Forest** | 0.4198 | 0.4513 | 0.4206 | 0.4167 | 13.03 s | 0.03 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **MLP (128)** | 0.5926 | 0.6440 | 0.5938 | 0.6024 | 102.76 s | **0.03 s** | $(648, 34596) \to (162, 34596)$ |

---

## 🔍 Comprehensive Discussion & Deep-Dive Findings

### 1. Fashion-MNIST: The Regime of Linearly Separable Manifolds

```
Accuracy Spectrum on Fashion-MNIST:
MLP           [===================================================] 86.20%
SVM           [================================================== ] 85.80%
Random Forest [================================================== ] 85.75%
KNN           [================================================== ] 85.55%
```

- **Analysis:** On Fashion-MNIST, the spread between the lowest-performing model (KNN at 85.55%) and the highest-performing model (MLP at 86.20%) is **less than 0.65%**.
- **Theoretical Insight:** Fashion-MNIST consists of isolated garments centered on uniform backgrounds. The spatial gradients captured by HOG provide strong linear class separability. When features cleanly partition class boundaries, **classifier complexity yields diminishing returns**.
- **Error Patterns:** Confusion matrices demonstrate that misclassifications are concentrated within semantically ambiguous garment shapes (e.g., `Shirt` vs. `T-shirt/top` vs. `Coat`), which share nearly identical gradient profiles regardless of the classifier used.

---

### 2. CIFAR-10: Complex Visual Context and Non-Linear Distributions

```
Accuracy Spectrum on CIFAR-10:
MLP           [===================================                 ] 47.80%
SVM           [===============================                     ] 43.75%
Random Forest [==============================                      ] 41.95%
KNN           [============================                        ] 39.90%
```

- **Analysis:** Accuracy drops significantly across all models (~40% - 48%). However, **MLP achieves a noticeable lead (+4.05% over SVM, +5.85% over RF, +7.90% over KNN)**.
- **Theoretical Insight:** CIFAR-10 images are low-resolution ($32 \times 32$), natural photographic scenes containing arbitrary backgrounds, scale variations, and camera viewpoints. Handcrafted gradient histograms (HOG) cannot capture rich color distributions or semantic contextual cues.
- **Why MLP Dominates:** Because the feature space is noisy and non-linearly distributed, the non-linear activation functions and hidden layer transformations of the MLP allow it to construct complex decision boundaries, outperforming linear hyperplanes and axis-aligned tree splits.

---

### 3. KTH-TIPS: The Curse of Dimensionality in Texture Classification

```
Accuracy Spectrum on KTH-TIPS (d = 34,596):
SVM           [===============================================     ] 64.81%
MLP           [==========================================          ] 59.26%
Random Forest [==============================                      ] 41.98%
KNN           [=================                                   ] 23.46% (COLLAPSED)
```

- **Analysis:** This dataset exhibits the most extreme performance divergence. **Linear SVM dominates with 64.81% accuracy and 70.89% precision**, whereas **KNN completely collapses to 23.46% accuracy and 15.61% Macro F1**.
- **Mathematical Explanation — The Curse of Dimensionality:**
  With $d = 34,596$ feature dimensions and only $N = 648$ training instances ($d \gg N$), the volume of the feature space grows exponentially. In ultra-high dimensional spaces, the ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1:
  $$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$
  Consequently, Euclidean distance loses discriminative power, rendering KNN's nearest neighbor queries nearly arbitrary.
- **Why Linear SVM Excels:**
  Linear SVM relies on convex quadratic programming to find the maximum-margin hyperplane:
  $$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
  In $d \gg N$ settings, data points are almost always linearly separable. SVM finds a global optimum without being affected by distance concentration, achieving high generalization despite severe sample scarcity.

---

### 4. Computational Complexity: Training Latency vs. Inference Speed

```
+---------------+------------------------+------------------------+
| Classifier    | Training Wall-Clock    | Inference Wall-Clock   |
+---------------+------------------------+------------------------+
| KNN           | O(1) [0.02s - 0.04s]   | O(N*d) [0.46s - 4.04s] |
| Random Forest | O(T * N * log N * d')  | O(T * depth) [0.03s]   |
| SVM (Linear)  | O(N^2 * d) [12s - 71s] | O(N_sv * d) [4s - 13s] |
| MLP           | O(E * N * d * h) [28s] | O(d * h) [< 0.01s]     |
+---------------+------------------------+------------------------+
```

- **Inference Bottlenecks (KNN):** While KNN requires virtually zero training time ($0.02$s), its testing time is dramatically higher because it must compute $O(N \cdot d)$ pairwise distances for every query. On Fashion-MNIST test data ($N=2000$), KNN takes **2.73s**, whereas MLP takes only **0.008s** (**>300x faster inference**).
- **Production Trade-offs:** For real-time embedded or edge systems, **MLP and Random Forest** provide superior deployment characteristics: although they require training upfront, their inference latency is measured in fractions of a millisecond.

---

## 🖼 Visualizations & Confusion Matrices

### Metric Comparison Charts

| Metric | Fashion-MNIST | CIFAR-10 | KTH-TIPS |
| :--- | :---: | :---: | :---: |
| **Accuracy** | ![Fashion Accuracy](results/figures/fashion_mnist_accuracy.png) | ![CIFAR10 Accuracy](results/figures/cifar10_accuracy.png) | ![KTH Accuracy](results/figures/kth_tips_accuracy.png) |
| **Macro F1** | ![Fashion F1](results/figures/fashion_mnist_macro_f1.png) | ![CIFAR10 F1](results/figures/cifar10_macro_f1.png) | ![KTH F1](results/figures/kth_tips_macro_f1.png) |
| **Macro Precision** | ![Fashion Precision](results/figures/fashion_mnist_precision.png) | ![CIFAR10 Precision](results/figures/cifar10_precision.png) | ![KTH Precision](results/figures/kth_tips_precision.png) |
| **Macro Recall** | ![Fashion Recall](results/figures/fashion_mnist_recall.png) | ![CIFAR10 Recall](results/figures/cifar10_recall.png) | ![KTH Recall](results/figures/kth_tips_recall.png) |
| **Train Time** | ![Fashion Train Time](results/figures/fashion_mnist_train_time.png) | ![CIFAR10 Train Time](results/figures/cifar10_train_time.png) | ![KTH Train Time](results/figures/kth_tips_train_time.png) |
| **Test Time** | ![Fashion Test Time](results/figures/fashion_mnist_test_time.png) | ![CIFAR10 Test Time](results/figures/cifar10_test_time.png) | ![KTH Test Time](results/figures/kth_tips_test_time.png) |

---

### Confusion Matrices

#### 1. Fashion-MNIST Confusion Matrices
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![Fashion SVM CM](results/confusion_matrices/fashion_mnist_svm_cm.png) | ![Fashion KNN CM](results/confusion_matrices/fashion_mnist_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![Fashion RF CM](results/confusion_matrices/fashion_mnist_random_forest_cm.png) | ![Fashion MLP CM](results/confusion_matrices/fashion_mnist_mlp_cm.png) |

#### 2. CIFAR-10 Confusion Matrices
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![CIFAR SVM CM](results/confusion_matrices/cifar10_svm_cm.png) | ![CIFAR KNN CM](results/confusion_matrices/cifar10_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![CIFAR RF CM](results/confusion_matrices/cifar10_random_forest_cm.png) | ![CIFAR MLP CM](results/confusion_matrices/cifar10_mlp_cm.png) |

#### 3. KTH-TIPS Confusion Matrices
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![KTH SVM CM](results/confusion_matrices/kth_tips_svm_cm.png) | ![KTH KNN CM](results/confusion_matrices/kth_tips_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![KTH RF CM](results/confusion_matrices/kth_tips_random_forest_cm.png) | ![KTH MLP CM](results/confusion_matrices/kth_tips_mlp_cm.png) |

---

## 💡 Synthesis: Validation of the 'No Free Lunch' Theorem

This study provides clean empirical evidence validating Wolpert's **No Free Lunch Theorem** for pattern recognition:

$$\text{Performance} = f(\text{Feature Inductive Bias}, \text{Classifier Inductive Bias}, \text{Data Manifold Geometry})$$

```
+--------------------------------------------------------------------------------------------------+
|                                    DECISION MATRIX FOR PRACTITIONERS                             |
+-------------------+-----------------------------------+------------------------------------------+
| Data Regime       | Recommended Model                 | Primary Theoretical Justification        |
+-------------------+-----------------------------------+------------------------------------------+
| High Dimensional  | Linear Support Vector Machine     | Convex margin maximization; immune to    |
| (d >> N, Texture) | (SVM)                             | metric collapse and curse of dimensional.|
+-------------------+-----------------------------------+------------------------------------------+
| Complex / Non-Lin | Multi-Layer Perceptron            | Non-linear universal approximation;      |
| (Object Scenes)   | (MLP)                             | builds hierarchical internal features.   |
+-------------------+-----------------------------------+------------------------------------------+
| Clean Silhouette  | Random Forest / Linear SVM /      | Features already linearly separate classes|
| (Structured Data) | KNN (Any standard baseline)       | cleanly; model complexity is redundant.  |
+-------------------+-----------------------------------+------------------------------------------+
| Low-Latency Edge  | Multi-Layer Perceptron /          | Fast forward-pass matrix multiplication; |
| Inference         | Random Forest                     | sub-millisecond evaluation latency.      |
+-------------------+-----------------------------------+------------------------------------------+
```

---

## 📁 Project Architecture & Repository Structure

```
pr-classifier-project/
│
├── config.py                     # Central configuration (paths, limits, constants)
├── main.py                       # Single-command pipeline execution entrypoint
├── requirements.txt              # Core Python dependency specifications
├── .gitignore                    # Production git rules excluding caches and raw bulk data
├── README.md                     # Main English documentation & research report
├── README_TR.md                  # Comprehensive Turkish documentation
│
├── src/                          # Modular Python source packages
│   ├── __init__.py
│   ├── classifiers.py            # Instantiation of SVM, KNN, RF, and MLP models
│   ├── data_loader.py            # Ingestion, grayscale conversion, and normalization
│   ├── features.py               # Standardized HOG feature extraction routines
│   ├── train_eval.py             # Fitting, inference timing, and metric calculation
│   ├── visualization.py          # Bar charts and annotated confusion matrix plots
│   └── experiments.py            # Orchestrator running benchmark loops
│
├── data/                         # Data directory structure
│   ├── raw/
│   │   └── KTH_TIPS/             # 10 texture classes (810 images)
│   └── processed/                # Placeholder for cached intermediate tensors (.gitkeep)
│
├── results/                      # Generated experimental artifacts
│   ├── metrics/
│   │   └── classifier_summary_results.csv   # Consolidated CSV benchmark matrix
│   ├── figures/                  # 18 bar charts (Accuracy, F1, Latencies per dataset)
│   └── confusion_matrices/       # 12 high-resolution annotated confusion matrices
│
├── notebooks/                    # Interactive exploratory Jupyter Notebooks
│   ├── 01_data_check.ipynb
│   ├── 02_hog_preview.ipynb
│   └── 03_result_analysis.ipynb
```

---

## 🚀 Installation & Reproducibility Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/pr-classifier-project.git
cd pr-classifier-project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Full Benchmark Pipeline
```bash
python main.py
```
*The script will automatically:*
1. Ingest Fashion-MNIST, CIFAR-10, and KTH-TIPS datasets.
2. Extract HOG feature vectors for all train/test splits.
3. Train and evaluate SVM, KNN, Random Forest, and MLP models.
4. Export the summary results to `results/metrics/classifier_summary_results.csv`.
5. Render and save all metric comparison bar charts into `results/figures/`.
6. Generate and save annotated confusion matrices into `results/confusion_matrices/`.

---

## 📑 Academic Presentation & Defense Overview

This research study was originally presented as a comprehensive 37-slide academic oral defense during undergraduate studies for the **Pattern Recognition** course. The full presentation structure and topical distribution is summarized below:

### Slide Deck Architecture & Topical Syllabus:
- **Slides 1–6:** Research Overview, Problem Statement, Hypotheses, and HOG Experimental Control Strategy
- **Slides 7–12:** Dataset Profiling, Inductive Biases, Classifier Mathematics, and Evaluation Metric Definitions
- **Slides 13–20:** Experiment 1: Fashion-MNIST Predictions, Metric Charts, and Confusion Matrix Diagnostics
- **Slides 21–27:** Experiment 2: CIFAR-10 Expectations, Deep Analysis, and Model Capacity Insights
- **Slides 28–34:** Experiment 3: KTH-TIPS Texture Analysis, Metric Collapse, and the Curse of Dimensionality
- **Slides 35–37:** Cross-Dataset Key Findings, Algorithmic Synthesis, and Conclusion

---

## 📜 Citation & Attribution

If you use or reference this benchmark, code, or experimental methodology in academic research or technical studies, please cite it as:

```bibtex
@misc{sandal2026prclassifier,
  author       = {Ersin Sandal},
  title        = {Quantitative Comparative Analysis of Classification Methods in Pattern Recognition: HOG-based Evaluation of Diverse Classifiers},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/<your-username>/pr-classifier-project}},
  note         = {Undergraduate Academic Research Coursework in Pattern Recognition}
}
```

---

<div align="center">
  <sub>Developed and maintained with precision by <b>Ersin Sandal</b>. Released under the MIT License.</sub>
</div>
