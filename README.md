# Heart Disease Prediction

A professional machine learning project to predict the presence of heart disease in patients using clinical parameters. The project implements exploratory data analysis (EDA), processes patient data, compares multiple classification models, and provides detailed performance evaluation including feature importance and confusion matrix.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Details](#dataset-details)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Models Compared](#models-compared)
- [Performance Results](#performance-results)
- [Key Insights & Feature Importance](#key-insights--feature-importance)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

## 🔍 Overview
Heart disease is one of the leading causes of mortality worldwide. Early and accurate diagnosis of heart conditions can significantly improve patient outcomes. This project leverages machine learning to build a predictive classification pipeline. It performs Exploratory Data Analysis (EDA) on clinical records, scales features, and evaluates six different classification algorithms under a 5-fold cross-validation scheme to find the most robust predictive model.

## 📊 Dataset Details
The dataset used is `heart.csv`, which contains clinical data for patients with the following features:

| Feature | Description |
|---|---|
| **age** | Age of the patient in years |
| **sex** | Gender of the patient (1 = male, 0 = female) |
| **cp** | Chest pain type (4 values: 0, 1, 2, 3) |
| **trestbps** | Resting blood pressure (in mm Hg on admission to the hospital) |
| **chol** | Serum cholesterol in mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| **restecg** | Resting electrocardiographic results (values 0, 1, 2) |
| **thalach** | Maximum heart rate achieved |
| **exang** | Exercise induced angina (1 = yes; 0 = no) |
| **oldpeak** | ST depression induced by exercise relative to rest |
| **slope** | The slope of the peak exercise ST segment |
| **ca** | Number of major vessels (0-3) colored by fluoroscopy |
| **thal** | Thalassemia status (3 = normal; 6 = fixed defect; 7 = reversable defect) |
| **target** | Diagnosis of heart disease (1 = presence, 0 = absence) |

---

## ⚙️ System Architecture & Workflow
1. **Exploratory Data Analysis (EDA)**:
   - Correlation Heatmap to inspect relationships between variables.
   - Distribution of the Target Class to check for class balance.
   - Age Distribution grouped by heart disease presence.
   - Chest Pain (cp) impact on the target variable.
2. **Data Preprocessing**:
   - Feature-Target separation.
   - Standard Scaling of features using `StandardScaler` to ensure scale-sensitive models (like SVM and KNN) perform optimally.
3. **Model Evaluation & Selection**:
   - Train-Test split (80/20) for standard validation.
   - 5-Fold Cross Validation across the entire scaled dataset.
4. **Post-Training Diagnostics**:
   - Feature Importance analysis (using Random Forest).
   - Confusion Matrix and Classification Report (Precision, Recall, F1-Score).

---

## 🤖 Models Compared
The pipeline trains and compares the following classification models:
* **Logistic Regression** (max_iter=103)
* **Decision Tree** (max_depth=3)
* **Random Forest** (n_estimators=100)
* **Support Vector Machine (SVM)** (RBF kernel)
* **Gaussian Naive Bayes**
* **K-Nearest Neighbors (KNN)** (n_neighbors=5)

---

## 📈 Performance Results

### 1. Cross-Validation Results (5-Fold CV Accuracy)
Models are evaluated using 5-fold cross-validation on the scaled dataset:

| Model Name | Mean CV Accuracy |
| :--- | :---: |
| **Random Forest** | **99.71%** |
| **SVM (RBF)** | **92.20%** |
| **Logistic Regression** | **84.59%** |
| **K-Nearest Neighbors** | **83.32%** |
| **Decision Tree** | **83.02%** |
| **Gaussian Naive Bayes** | **82.15%** |

### 2. Random Forest Test Set Performance (80/20 Split)
* **Test Accuracy**: **98.54%**
* **Confusion Matrix**:
  ```
  [[102    0]
   [  3  100]]
  ```
* **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.97      1.00      0.99       102
             1       1.00      0.97      0.99       103

      accuracy                           0.99       205
     macro avg       0.99      0.99      0.99       205
  weighted avg       0.99      0.99      0.99       205
  ```

---

## 🔑 Key Insights & Feature Importance
The Random Forest Classifier identified the following clinical features as the most critical predictors for heart disease:

1. **cp** (Chest pain type): **13.51%**
2. **ca** (Number of major vessels colored by fluoroscopy): **12.73%**
3. **thalach** (Maximum heart rate achieved): **12.22%**
4. **oldpeak** (ST depression induced by exercise): **12.19%**
5. **thal** (Thalassemia): **11.05%**

*Clinical Note: Chest pain type (`cp`) and the number of major vessels (`ca`) carry the highest weight in determining prediction outcomes.*

---

## 🚀 Installation & Usage

### Prerequisites
Make sure you have Python 3.8+ installed.

### Dependencies
Install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Ravi20051/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Run the main prediction script:
   ```bash
   python heart_disease_prediction.py
   ```
