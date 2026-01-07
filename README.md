# Employee Attrition Prediction

This project implements a machine learning pipeline to predict employee attrition based on synthetic organizational data. It explores demographic, professional, and satisfaction-related factors to identify employees at high risk of leaving.

##  Project Overview

The goal is to provide a predictive model that helps HR departments proactively address turnover. The project covers the full data science lifecycle:

* **Synthetic Data Generation:** Creating a custom dataset with 1,000 records.
* **EDA:** Visualizing class distributions and feature correlations.
* **Preprocessing:** Handling categorical encoding and feature scaling.
* **Modeling:** Comparing Logistic Regression and Random Forest Classifiers.
* **Tuning:** Optimizing model parameters using GridSearchCV.

---

##  Dataset Specifications

The model uses a synthetic dataset (`synthetic_employee_attrition.csv`) containing 14 features:

* **Demographics:** Age, Gender.
* **Professional:** Department, Job Role, Monthly Income, Years at Company.
* **Behavioral/Satisfaction:** Job Satisfaction, Overtime, Work-Life Balance, Environment Satisfaction, Performance Rating.
* **Target:** `Attrition` (Yes/No).

---

##  Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/employee-attrition-prediction.git
cd employee-attrition-prediction

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```



### Running the Project

* Execute the Jupyter Notebook to see the analysis and training steps:
```bash
jupyter notebook attrition_prediction.ipynb

```


* The final trained model will be saved as `attrition_model.pkl`.

---

##  Model Performance

| Model | Accuracy | Observations |
| --- | --- | --- |
| **Logistic Regression** | 100% | Performed exceptionally well on the linear logic of the synthetic data. |
| **Random Forest** | 99% | Struggled with minority class recall due to extreme class imbalance. |

> **Note:** The high accuracy is a result of the specific rules defined in the synthetic data generator. In a real-world scenario, features would be more stochastic and noisy.

---

##  Built With

* **Pandas & NumPy** - Data manipulation.
* **Seaborn & Matplotlib** - Data visualization.
* **Scikit-Learn** - Machine learning modeling and preprocessing.
* **Joblib** - Model serialization.

---

##  Future Improvements

* **Address Class Imbalance:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to improve Random Forest performance on "Attrition = Yes" cases.
* **Feature Engineering:** Create new features like "Income per Year of Experience."
* **Real-world Data:** Validate the pipeline using the [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

---

### Would you like me to help you write the code for the SMOTE implementation to fix those Random Forest warnings?
