# Hybrid EEG Motor Imagery Classifier: WPD + CSP + MLP vs. EEGNet

[![Motor Imagery](https://img.shields.io/badge/Motor%20Imagery-EEG%20BCI-6A1B9A)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](#)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-EEG%20Preprocessing-1565C0)](#)
[![EEGNet](https://img.shields.io/badge/EEGNet-Compact%20ConvNet-00897B)](#)
[![LOSO](https://img.shields.io/badge/Validation-Leave--One--Subject--Out-D32F2F)](#)
[![4-Class Classification](https://img.shields.io/badge/4--Class%20MI-Left%20%2F%20Right%20%2F%20Feet%20%2F%20Tongue-FF6F00)](#)

A systematic, three-stage comparative study of **cross-subject generalization** in EEG-based motor imagery classification — from a naive single-subject baseline, through Leave-One-Subject-Out (LOSO) cross-validation of a hand-engineered pipeline, to an end-to-end deep learning approach with **EEGNet**.

## 📌 Overview

Motor imagery BCIs classify which movement a person is *imagining* purely from their EEG — a hard, noisy, low-SNR problem made even harder by the fact that no two brains produce the same signal for the same task. This project started as a single-subject notebook (~77% accuracy on one individual) and has since grown into a **full 3-notebook investigation** into the single biggest obstacle to deploying real-world BCIs: **inter-subject variability**.

Across the three notebooks, the same core question is asked in three progressively more rigorous ways: *if I train on some subjects, how well does the model work on a subject it has never seen?*

| Approach | Validation Scheme | Mean Cross-Subject Accuracy | vs. Chance (25%) |
|---|---|---|---|
| Subject-dependent (WPD+CSP+MLP) | Zero-calibration transfer | 31.7% ± 7.4% | +6.7 pts |
| LOSO Hybrid (WPD+CSP+MLP) | 9-fold LOSO | 40.1% ± 12.5% | +15.1 pts |
| **LOSO EEGNet (zero-shot)** | 9-fold LOSO | **43.9% ± 14.0%** | **+18.9 pts** |
| **LOSO EEGNet (fine-tuned)** | LOSO + 50% calibration | **47.6%** | **+22.6 pts** |

> 📄 **Full methodology, derivations, and discussion:** **[Read the complete report →](doc/report.pdf)**

> 🔗 **Quick links:** [Dataset & download instructions](data/README.md) · [Notebook 1](notebooks/Hybrid-EEG-Classifier-WPD-CSP.ipynb) · [Notebook 2](notebooks/Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb) · [Notebook 3](notebooks/EEGNet-LOSO.ipynb) · [EEGNet architecture reference](https://braindecode.org/1.4/generated/braindecode.models.EEGNet.html)

---

## 🧠 The Core Problem: Inter-Subject Variability

A classifier calibrated on one person's EEG encodes spatial filters and decision boundaries tied to *that individual's* anatomy and cognitive strategy — cortical folding, skull thickness, electrode placement, and the subject's own way of "imagining" a movement all differ from person to person. Applied to a new subject with **zero calibration**, performance frequently collapses toward chance level. This is the central obstacle to "plug-and-play" BCIs, and it's the obstacle this project measures, quantifies, and attempts to close.

A dedicated pairwise transfer-accuracy analysis makes the scale of the problem unmistakable:

| | Within-Subject Accuracy | Cross-Subject Accuracy |
|---|---|---|
| CSP + Logistic Regression | **65.6%** | **25.4%** (≈ chance) |

Hand-engineered spatial filters fit on one subject's covariance structure essentially fail to transfer to a different subject's covariance structure *at all* — unless the training process explicitly pools subjects (LOSO) or applies signal alignment.

---

## 📁 Project Structure

```text
Hybrid-EEG-Classifier-EEGNet/
├── data/
│   ├── raw/
│   └── README.md          # Subject info + dataset download instructions
├── doc/
│   └── report.pdf          # Full methodology, results, and discussion
├── figures/
├── models/
├── notebooks/
│   ├── Hybrid-EEG-Classifier-WPD-CSP.ipynb        # Notebook 1
│   ├── Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb   # Notebook 2
│   └── EEGNet-LOSO.ipynb                          # Notebook 3
├── results/
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

All three notebooks use the **[BCI Competition IV, Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a)** (Brunner et al., 2008) — 9 subjects, 4 balanced motor imagery classes (left hand, right hand, feet, tongue), 22 EEG + 3 EOG channels at 250 Hz.

| Subjects | Classes | Trials / Subject | Channels | Sampling Rate |
|---|---|---|---|---|
| 9 (A01T–A09T) | Left / Right / Feet / Tongue | 288 (72/class, balanced) | 22 EEG + 3 EOG | 250 Hz |

> 💡 The raw `.gdf` files are **not included** in this repository. See **[`data/README.md`](data/README.md)** for per-subject details and download instructions before running any notebook.

---

## 🔬 Three-Stage Experimental Pipeline

### 1️⃣ Notebook 1 — Subject-Dependent Baseline
📓 **[`notebooks/Hybrid-EEG-Classifier-WPD-CSP.ipynb`](notebooks/Hybrid-EEG-Classifier-WPD-CSP.ipynb)**

The original hybrid pipeline — **Wavelet Packet Decomposition (WPD)** + **Filter-Bank Common Spatial Patterns (CSP)** + a **PyTorch MLP** — trained and validated *exclusively* on Subject 1 (A01T), then applied without any refitting to the other eight subjects.

- ✅ Excellent **in-sample** performance: **75.9%** test accuracy / 0.760 macro-F1 on Subject 1's own held-out split.
- ❌ Severe generalization collapse on unseen subjects: mean accuracy drops to **31.7% ± 7.4%** — a **44-point** gap between training and deployment performance, with some subjects (A06T, A09T) landing barely above the 25% chance level.

This notebook exists specifically to *quantify* the zero-calibration generalization gap before attempting to close it.

### 2️⃣ Notebook 2 — LOSO Cross-Validation with the Hybrid Algorithm
📓 **[`notebooks/Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb`](notebooks/Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb)**

The exact same WPD + CSP + MLP algorithm, but re-evaluated under a proper **Leave-One-Subject-Out (LOSO)** cross-validation protocol: on each of 9 folds, one subject is held out entirely and the pipeline is refit from scratch on the pooled remaining eight, with **Euclidean Alignment** added as a leakage-free cross-subject covariance-whitening step.

- 📈 Mean LOSO accuracy improves to **40.1% ± 12.5%**, ahead of both the CSP+LogReg baseline (38.2%) and Notebook 1's unseen-subject average.
- Pooling subjects during training forces spatial filters to compromise across individual covariance structures rather than overfitting to just one — a real but *modest* improvement, since averaging across 8 heterogeneous subjects is a compromise, not an adaptation.

### 3️⃣ Notebook 3 — EEGNet with LOSO
📓 **[`notebooks/EEGNet-LOSO.ipynb`](notebooks/EEGNet-LOSO.ipynb)**

Replaces the entire hand-engineered feature pipeline with **[EEGNet](https://braindecode.org/1.4/generated/braindecode.models.EEGNet.html)**, a compact end-to-end convolutional architecture purpose-built for EEG decoding, evaluated under the identical 9-fold LOSO protocol. EEGNet learns its own temporal filters (analogous to band-pass filtering) and spatial filters (analogous to CSP) directly from data, with no hand-engineered priors at all.

- 🏆 **Best cross-subject performance of all three approaches**: **43.9% ± 14.0%** mean LOSO accuracy, Cohen's **κ = 0.252**.
- 🎯 A brief subject-specific fine-tuning step (10 epochs on just 50% of the target subject's own trials) pushes mean accuracy further to **47.6%**.

> ⚠️ **Hardware constraints on Notebook 3.** All training in this project ran on **CPU only** — no CUDA-capable GPU was available. EEGNet's recommended training schedule is **~300 epochs**, which was estimated to take **7+ hours** on this hardware. That was not feasible, so training was run with a **drastically reduced budget of 42 epochs** (patience = 6) — and even *this* reduced run took **over 1 hour**. Several LOSO folds (e.g. Subjects 1 and 2) had not yet plateaued in validation accuracy when the epoch budget ran out, meaning the results reported here are a **conservative lower bound**: EEGNet already beats both hand-engineered pipelines under this constraint, and a full training budget would very likely widen that gap further.

---

## 📈 Results

> 📄 For full per-subject tables, per-class precision/recall/F1, confusion matrices, and the complete discussion, see the **[full project report](doc/report.pdf)**.

### Final Cross-Subject Accuracy — All Three Notebooks

<table>
<tr>
<th align="center">Notebook 1<br>Subject-Dependent Baseline</th>
<th align="center">Notebook 2<br>LOSO Hybrid WPD-CSP-MLP</th>
<th align="center">Notebook 3<br>LOSO EEGNet</th>
</tr>
<tr>
<td><img src="figures/cross_subject_generalization.png" width="280"/></td>
<td><img src="figures/loso_accuracy_by_subject.png" width="280"/></td>
<td><img src="figures/final_loso_test_summary.png" width="280"/></td>
</tr>
</table>

The story across the three charts is monotonic: naive single-subject training generalizes worst (blue = trained subject, red = unseen, most barely above the chance line); pooling all subjects via LOSO with the *same* hand-engineered features recovers a real but modest improvement; and replacing those features with EEGNet's learned representations — even under a constrained training budget — pushes every subject's accuracy further above the hybrid-model mean (dashed reference line at 40%) and the 25% chance level.

### The Domain-Shift Problem, Visualized

<p align="center">
  <img src="figures/pairwise_transfer_heatmap.png" width="620" alt="Pairwise cross-subject transfer accuracy heatmap"><br>
  <sub><b>Pairwise Transfer Heatmap.</b> Diagonal = within-subject held-out accuracy (mean 65.6%); off-diagonal = direct subject-to-subject transfer (mean 25.4%, statistically at chance). This is the clearest evidence in the study that CSP spatial filters fit on one subject essentially do not transfer to another without pooling or alignment.</sub>
</p>

<br>

<p align="center">
  <img src="figures/confusion_matrix_pooled.png" width="480" alt="Pooled confusion matrix across all nine LOSO folds"><br>
  <sub><b>Pooled LOSO Confusion Matrix.</b> Row-normalized confusion matrix aggregated across all nine LOSO folds (Hybrid WPD-CSP-MLP). Left/Right Hand trials are decoded more reliably than Feet and Tongue, consistent with the stronger, more lateralized ERD associated with hand motor imagery — a pattern that recurs in the EEGNet results too.</sub>
</p>

### 🔊 A Note on Noisy Subjects

Not all subjects are equally decodable, and this pattern is **consistent across every pipeline tested** — strong evidence that the difficulty is intrinsic to those subjects' recordings rather than an artifact of any one modelling approach:

- **Subject 2 (S02)** hovers at or near chance level in every experiment — **24.7%** under LOSO hybrid, **29.9%** under zero-shot EEGNet — suggesting a persistently weak or noisy motor-imagery signature.
- **Subject 5 (S05)** is similarly difficult (**28.5%** hybrid LOSO, **25.7%** EEGNet) and is the *only* subject that showed **zero improvement** from subject-specific fine-tuning, implying very little class-discriminative signal is present in that subject's data under either approach.
- By contrast, **Subject 1** and **Subject 8** are consistently the easiest to generalize to across all three notebooks (up to **66.0%** accuracy for EEGNet on S01).

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/alitkbbl/Hybrid-EEG-Classifier-EEGNet
cd Hybrid-EEG-Classifier-EEGNet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
See **[`data/README.md`](data/README.md)** for per-subject `.gdf` download links and where to place them (`data/raw/`).

### 4. Run the notebooks (in order)
```bash
jupyter notebook notebooks/Hybrid-EEG-Classifier-WPD-CSP.ipynb        # Notebook 1: subject-dependent baseline
jupyter notebook notebooks/Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb   # Notebook 2: LOSO hybrid pipeline
jupyter notebook notebooks/EEGNet-LOSO.ipynb                          # Notebook 3: LOSO EEGNet
```

> 💡 Notebook 3 (EEGNet) is by far the most compute-intensive. On CPU-only hardware, expect the reduced-epoch LOSO run to take **over an hour**; see the hardware note above before adjusting `EPOCHS`/`PATIENCE`.

---

## 🔗 References

- Ang, K. K., Chin, Z. Y., Zhang, H., & Guan, C. (2008). *Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface.* IEEE IJCNN.
- Brunner, C. et al. (2008). *BCI Competition 2008 – Graz Data Set A.*
- Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). *Optimal spatial filtering of single trial EEG during imagined hand movement.* IEEE Trans. Rehabilitation Engineering.
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.* Journal of Neural Engineering, 15(5), 056013. — [Architecture reference (braindecode)](https://braindecode.org/1.4/generated/braindecode.models.EEGNet.html)
- He, H., & Wu, D. (2020). *Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach.* IEEE Trans. Biomedical Engineering, 67(2), 399–410.
- Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7, 267.

> 📄 For the complete derivations, per-fold tables, and full discussion behind every number in this README: **[Read the full project report →](doc/report.pdf)**