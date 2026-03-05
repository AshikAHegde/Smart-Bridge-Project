md
# Predictive Pulse: Harnessing Machine Learning for Blood Pressure Analysis

## Project Overview
Predictive Pulse is a machine learning-based system designed to predict hypertension risk using patient health data. The project applies data preprocessing, exploratory data analysis, model training, evaluation, and deployment through a web application. The goal is to assist in early detection of hypertension risk using machine learning models.

---

# Project Structure



Smart-Bridge-Project
│
├ Dataset
│   ├ patient_data.xlsx
│   └ patient_data_cleaned.xlsx
│
├ static
│   └ style.css
│
├ templates
│   └ index.html
│
├ visualizations_output
│   ├ 01_gender_distribution.png
│   ├ 02_gender_distribution.png
│   ├ 03_hypertension_stages.png
│   ├ 04_correlation_heatmap.png
│   ├ 05_medication_vs_stage.png
│   ├ 06_age_vs_stages.png
│   ├ 07_pairplot_systolic.png
│   ├ 08_severity_distribution.png
│   ├ 09_systolic_distribution.png
│   └ 10_diastolic_distribution.png
│
├ app.py
├ train_model.py
├ data_analysis.py
├ visualizations.py
├ logreg_model.pkl
├ README.md



---

# Milestone 1: Data Collection and Preparation

## Dataset
The dataset contains patient health records including demographic and physiological attributes used for hypertension prediction.

### Key Attributes
- Age
- Gender
- BMI
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Heart Rate
- Glucose Level
- Medication Status
- Hypertension Stage

### Data Cleaning
The following preprocessing steps were performed:

- Handling missing values
- Removing duplicate entries
- Standardizing column names
- Encoding categorical variables
- Saving the cleaned dataset as `patient_data_cleaned.xlsx`

---

# Milestone 2: Exploratory Data Analysis (EDA)

EDA was performed to understand relationships between features and hypertension stages.

### Visualizations Generated

1. Gender distribution
2. Hypertension stage distribution
3. Correlation heatmap
4. Medication vs hypertension stage
5. Age vs hypertension stage
6. Pairplot of systolic pressure
7. Severity distribution
8. Systolic blood pressure distribution
9. Diastolic blood pressure distribution

These visualizations help identify important features affecting hypertension prediction.

---

# Milestone 3: Model Building

## Data Splitting

The dataset was divided into training and testing sets using the train-test split technique.

- Training Data: 80%
- Testing Data: 20%
- Stratified sampling was used to maintain class balance.

## Algorithms Implemented

Seven machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Ridge Classifier
7. Gaussian Naive Bayes

### Model Performance Summary

| Model | Accuracy |
|------|---------|
| Logistic Regression | 95.2% |
| Decision Tree | 100% |
| Random Forest | 100% |
| SVM | 100% |
| KNN | 98.1% |
| Ridge Classifier | 90.0% |
| Gaussian Naive Bayes | 97.0% |

---

# Milestone 4: Model Selection and Overfitting Analysis

## Overfitting Analysis

Several models achieved perfect accuracy (100%), including:

- Decision Tree
- Random Forest
- Support Vector Machine

While high accuracy may appear ideal, it often indicates **overfitting**, where the model memorizes training data rather than learning general patterns.

### Risks of Overfitting

- Poor performance on unseen data
- Reduced reliability in clinical scenarios
- Incorrect predictions for new patients
- Safety risks in medical decision-making

## Why Logistic Regression Was Selected

Logistic Regression was selected as the final model because it demonstrated strong generalization ability with balanced performance.

### Key Reasons

- Stable accuracy (95.2%)
- Balanced precision and recall
- Lower risk of overfitting
- Interpretable model suitable for healthcare applications

### Key Performance Metrics

| Metric | Score |
|------|------|
| Overall Accuracy | 95.2% |
| Macro F1 Score | 0.95 |
| Weighted F1 Score | 0.95 |
| Crisis Recall | 100% |
| Stage-2 Precision | 100% |

---

# Milestone 5: Model Deployment

A Flask-based web application was developed to allow users to input patient health data and obtain hypertension risk predictions.

## Web Application Features

- User input form for health parameters
- Machine learning prediction engine
- Real-time hypertension risk prediction
- Simple and user-friendly interface

### Input Parameters

- Age
- BMI
- Heart Rate
- Glucose Level

### Output

The application predicts:

- Low Risk of Hypertension
- High Risk of Hypertension

---

# Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Flask
- HTML/CSS

---

# Conclusion

The Predictive Pulse system demonstrates how machine learning can assist in predicting hypertension risk using patient health data. Logistic Regression was selected as the final model due to its strong generalization performance and interpretability, making it suitable for healthcare applications. The deployed web application provides an accessible interface for real-time hypertension risk prediction.
