# Dataset

This project uses the **BCI Competition IV, Dataset 2a** for motor imagery EEG classification.

## 📁 Dataset Structure

The raw EEG recordings are stored in:

```text
data/
└── raw/
    ├── A01T.gdf
    ├── A01E.gdf
    ├── A02T.gdf
    ├── ...
    └── A09E.gdf
```

The recordings are provided in **GDF (General Data Format)**.

## ⬇️ Download

The dataset can be downloaded from Kaggle:

🔗 **[BCI IV-2a Raw GDF](https://www.kaggle.com/datasets/moonimint/bci-iv-2a-raw-gdf)**

After downloading, place the `.gdf` files inside:

```text
data/raw/
```

## ℹ️ Dataset Notes

* 🧑‍💻 **9 subjects** are included in the dataset.
* 🎯 Four motor imagery classes are used: **Left Hand, Right Hand, Feet, and Tongue**.
* ⚠️ **Subjects 02 and 05 are considerably noisier and more challenging** than the other subjects, which can result in lower classification performance.
* 🔬 The preprocessing and feature extraction pipeline operates directly on the raw **GDF** recordings.
