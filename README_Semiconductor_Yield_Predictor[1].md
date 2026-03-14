# Semiconductor Yield Predictor

A machine learning pipeline to predict semiconductor manufacturing yield from multi-sensor data - turning raw factory sensor readings into actionable manufacturing intelligence.

---

## Problem Statement

In semiconductor fabrication, a small drop in yield can cost millions. Sensors on the production line generate hundreds of measurements per wafer batch. This project builds an ML model that predicts whether a batch will pass or fail quality thresholds - before the process is complete - enabling early intervention.

---

## Dataset

- **Source:** SECOM Manufacturing Dataset (UCI Machine Learning Repository)
- **Size:** ~1,567 samples, 590 sensor features
- **Target:** Binary - Pass (1) / Fail (-1) quality outcome
- **Challenge:** High dimensionality, heavy class imbalance (~93% pass rate), missing values

---

## Approach

1. **EDA** — missing value analysis, class distribution, sensor correlation analysis
2. **Data Cleaning** — imputed missing values with column medians, dropped near-zero variance features
3. **Dimensionality Reduction:**
   - Variance threshold feature selection
   - PCA — reduced 590 features to ~230 components (60% reduction, 95% variance retained)
4. **Class Imbalance Handling** — SMOTE to oversample the minority (fail) class
5. **Model Training:**
   - Random Forest Classifier (best performer)
   - Logistic Regression (baseline)
6. **Evaluation** — Precision, Recall, F1-score (prioritised recall to minimise missed failures)

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 88%+ |
| Recall (Fail class) | 78% |
| Precision (Fail class) | 74% |
| Dimensionality reduction | 590 → 230 features (60% reduction) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| ML | Scikit-learn (Random Forest, PCA, SMOTE) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
Semi-Conductor-Yield-Predictor/
│
├── data/
│   ├── secom.data
│   └── secom_labels.data
├── notebooks/
│   └── semiconductor_yield.ipynb
├── outputs/
│   ├── pca_variance_plot.png
│   └── confusion_matrix.png
└── README.md
```

---

## Key Visualisations

- Explained variance ratio plot (PCA components)
- Class distribution before and after SMOTE
- Confusion matrix for the final model
- Top feature importances (pre-PCA sensor rankings)

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/manaswipriyaa/Semi-Conductor-Yield-Predictor.git
cd Semi-Conductor-Yield-Predictor

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter

# Launch the notebook
jupyter notebook notebooks/semiconductor_yield.ipynb
```

---

## Author

**Manaswi Priya Maddu**
B.Tech - AI & Machine Learning | Acharya Nagarjuna University
[LinkedIn](https://linkedin.com/in/manaswi-priya-2126481b8) | [GitHub](https://github.com/manaswipriyaa)
