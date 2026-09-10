# 📓 Notebooks

This folder contains the three notebooks developed for the **BCI Competition IV-2a** motor imagery EEG experiments.  
The notebooks follow the progression of the project from a subject-dependent baseline to cross-subject evaluation and end-to-end deep learning.

---

## 🧪 1. Hybrid WPD + CSP + MLP

**[`Hybrid-EEG-Classifier-WPD-CSP.ipynb`](Hybrid-EEG-Classifier-WPD-CSP.ipynb)**

The first notebook implements the original hand-engineered pipeline:

> **EEG → Wavelet Packet Decomposition (WPD) → Filter-Bank CSP → MLP**

The model is trained on **Subject 1 (A01T)** and then applied to the other subjects without recalibration.  
This experiment provides a simple subject-dependent baseline and shows how strongly performance changes when the model is transferred to unseen subjects.

---

## 🔄 2. LOSO Hybrid Pipeline

**[`Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb`](Hybrid-EEG-Classifier-WPD-CSP-LOSO.ipynb)**

The same **WPD + CSP + MLP** pipeline is evaluated using **9-fold Leave-One-Subject-Out (LOSO)** cross-validation.

For each fold, one subject is kept completely unseen while the remaining eight subjects are used for training. The notebook also applies **Euclidean Alignment** to reduce differences between subjects before feature extraction.

> **EEG → Alignment → WPD → CSP → MLP → 4-class prediction**

This provides a more consistent evaluation of cross-subject generalization than the first notebook.

---

## 🧬 3. EEGNet with LOSO

**[`EEGNet-LOSO.ipynb`](EEGNet-LOSO.ipynb)**

The third notebook replaces the manually engineered WPD/CSP feature pipeline with **EEGNet**, a compact convolutional neural network designed for EEG decoding.

Instead of extracting features explicitly, EEGNet learns **temporal and spatial representations directly from the preprocessed EEG** and performs the final four-class classification end-to-end.

The notebook uses the same **9-fold LOSO** evaluation setting, allowing the end-to-end approach to be compared with the hybrid pipeline under the same cross-subject protocol.

### EEGNet Architecture

<p align="center">
  <img src="../figures/notebook3_pipeline.png" width="100%" alt="EEGNet architecture and preprocessing pipeline">
</p>

<div style="text-align: center;">

<sub><b>Figure.</b> Preprocessing pipeline and EEGNet architecture used in Notebook 3, showing the flow from raw EEG recordings through preprocessing and learned temporal/spatial representations to four-class motor imagery classification.</sub>

</div>

---

## 📊 Notebook Summary

| Notebook | Main Approach | Evaluation |
|---|---|---|
| **1. Hybrid WPD + CSP + MLP** | Hand-engineered features + MLP | Subject-dependent + zero-calibration transfer |
| **2. LOSO Hybrid** | WPD + CSP + MLP + Euclidean Alignment | 9-fold LOSO |
| **3. EEGNet LOSO** | End-to-end temporal + spatial learning | 9-fold LOSO + fine-tuning |

> 📄 **Full methodology, derivations, and discussion:** **[Read the complete report →](../doc/Report.pdf)**
