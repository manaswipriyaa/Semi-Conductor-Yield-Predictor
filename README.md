# Semiconductor Yield Prediction

## Project Overview
This project focuses on predicting the **Pass/Fail yield** of semiconductor production entities using sensor data collected from the manufacturing process.  
Each record represents a single unit measured across 591 sensors, with the target variable indicating whether it passed (-1) or failed (1) in testing.  

The primary goal is to **build an intelligent classification system** that identifies failing components early — reducing waste, improving process efficiency, and optimizing yield.

---

## Problem Context
Modern semiconductor fabrication involves thousands of continuous sensor readings.  
Not all signals are useful — many are redundant or noisy.  
By applying **feature analysis, data balancing, and machine learning**, this project identifies the most relevant signals for yield prediction.

---

## Dataset Description
- **File:** `signal-data.csv`  
- **Samples:** 1567  
- **Features:** 591 sensor readings per unit  
- **Target Variable:**  
  - `-1` → Pass  
  - `1` → Fail  

---

## Steps Taken: 

### 1. Data Cleaning
- Removed timestamp and irrelevant columns.  
- Imputed missing values with median/mode.  
- Handled multicollinearity using correlation analysis.  

### 2. Exploratory Data Analysis (EDA)
- Visualized target class imbalance.  
- Examined key sensor distributions.  
- Generated correlation heatmap to identify redundant sensors.  
- Performed PCA for dimensionality understanding.  

### 3. Data Preprocessing
- Segregated features (X) and target (y).  
- Standardized numeric features.  
- Balanced the dataset using **SMOTE (Synthetic Minority Oversampling Technique)**.

### 4. Model Building
Trained and compared the following supervised models:
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Naive Bayes Classifier**

### 5. Model Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

**Key Finding:**  
Although SVM showed slightly higher overall accuracy, it failed to detect most “Fail” samples due to class imbalance.  
After SMOTE balancing, **Random Forest** achieved the best **F1 and recall** for the minority class, making it the ideal choice for this application.

---

## Final Model
**Model Chosen:** Random Forest Classifier  
**Reason:**  
- Strong recall and balanced F1 for both classes.  
- Handles correlated, high-dimensional features efficiently.  
- Provides interpretable feature importance for sensor diagnostics.

**Best Achieved Performance:**  
- Accuracy: ~93%  
- F1 (Fail): ~0.96  
- Recall (Fail): ~1.00  

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn  
- **Environment:** Google Colab  

---

## Project Files
- `Major_Project.ipynb` → Main notebook with full implementation  
- `signal-data.csv` → Dataset  
- `final_model.pkl` → Trained Random Forest model  
- `README.txt` → Project documentation  

---

## Key Takeaway
This project demonstrates the power of **machine learning in semiconductor manufacturing** — leveraging data-driven insights to improve yield, reduce costs, and strengthen process reliability.  
It also highlights the importance of **handling data imbalance** and **model interpretability** in real-world applications.

---

## Author 
Manaswi Priya Maddu

---
