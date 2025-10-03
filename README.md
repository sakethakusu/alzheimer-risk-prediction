# Alzheimer-Risk-Prediction-Using-ML
This project uses machine learning techniques to predict the onset of Alzheimer's Disease based on cognitive assessments, lifestyle habits, and medical history. Our approach focuses on early detection to enable better patient outcomes and informed decision-making.

##  Project Overview

- **Title:** From Cognitive Assessments to Lifestyle Data: A Predictive Model for Alzheimer’s Disease Onset
- **Team Members:** Saketha Kusu, Suhani Yalaga
- **Goal:** Develop a machine learning model to predict Alzheimer’s diagnosis using structured clinical and lifestyle data.
- **Dataset:** Kaggle Alzheimer’s Dataset (2,149 records, 35 features)

##  Dataset Details

| Category            | Features Included |
|---------------------|------------------|
| Demographics        | Age, Gender, Ethnicity, Education |
| Lifestyle Factors   | BMI, Smoking, Alcohol, Diet, Sleep, Physical Activity |
| Medical History     | Family History, Cardiovascular Disease, Diabetes, Depression |
| Clinical Measurements | BP, Cholesterol (LDL, HDL, Total), Triglycerides |
| Cognitive Assessments | MMSE, Functional Assessment, Memory Complaints |
| Symptoms            | Confusion, Disorientation, Forgetfulness, ADL issues |

- **Target Column:** `Diagnosis` (0 = No Alzheimer's, 1 = Alzheimer’s)
## Methods Used

1. **Data Preprocessing**
   - Handling missing values (mean/mode)
   - Removing duplicates
   - Label encoding categorical variables

2. **Feature Scaling and Selection**
   - StandardScaler for normalization
   - RFE (Recursive Feature Elimination) to select top 10 features

3. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)

4. **Model Training & Evaluation**
   - Algorithms: Logistic Regression, SVM, Random Forest
   - Metrics: Accuracy, Precision, Recall, F1-Score

5. **Hyperparameter Tuning**
   - GridSearchCV on Random Forest
   - Tested 216 combinations over 5 folds

6. **Patient Prediction Test**
   - Created a synthetic patient profile using top features
   - Model showed 90% confidence in classification

## Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Random Forest       | **0.91** | 0.94      | 0.86   | 0.90     |
| Logistic Regression | 0.85     | 0.87      | 0.77   | 0.81     |
| SVM                 | 0.83     | 0.85      | 0.74   | 0.79     |

**Best Model:** Random Forest after hyperparameter tuning

##  Future Work

- Integrate time-series analysis for disease progression
- Add multimodal data (MRI, genetics)
- Explore CNNs, LSTM, NLP on doctor notes
- Build a clinical decision support system with SHAP/LIME interpretability

## Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib / Seaborn for visualization
- Jupyter Notebook
- SMOTE for balancing
- GridSearchCV for tuning

## Youtube link:  https://youtu.be/w3NqzkMJKy0
