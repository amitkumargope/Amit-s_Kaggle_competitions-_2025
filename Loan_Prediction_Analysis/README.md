# 🏦 Loan Payback Prediction - Kaggle Competition

## 📋 Project Overview

This project provides a comprehensive analysis and prediction system for loan payback prediction using machine learning. The notebook includes detailed exploratory data analysis (EDA), feature engineering, multiple ML model comparisons, and interactive dashboards designed for stakeholders to make data-driven decisions.

## 🎯 Objective

Predict whether a loan will be paid back based on borrower characteristics and loan details. This is a binary classification problem where:
- **0**: Loan Not Paid Back
- **1**: Loan Paid Back

## 📊 Dataset Information

**Source**: Kaggle Competition - Loan Prediction Challenge

### Training Data
- **Size**: 593,994 samples
- **Features**: 13 variables (12 features + 1 target)
- **File**: `train.csv`

### Test Data
- **Size**: 254,571 samples
- **Features**: 12 variables (no target variable)
- **File**: `test.csv`

### Features Description

#### Numerical Features
1. **annual_income**: Annual income of the borrower ($)
2. **debt_to_income_ratio**: Ratio of debt to income (0-1 scale)
3. **credit_score**: Credit score of the borrower (300-850)
4. **loan_amount**: Amount of loan requested ($)
5. **interest_rate**: Interest rate on the loan (%)

#### Categorical Features
6. **gender**: Gender of the borrower (Male, Female, Other)
7. **marital_status**: Marital status (Single, Married, Divorced)
8. **education_level**: Education level (High School, Bachelor's, Master's, PhD, Other)
9. **employment_status**: Employment status (Employed, Unemployed, Self-employed, Student)
10. **loan_purpose**: Purpose of the loan (Debt consolidation, Business, Education, Home, Car, Medical, Vacation, Other)
11. **grade_subgrade**: Loan grade/subgrade (A1-E5)

#### Target Variable
12. **loan_paid_back**: Whether the loan was paid back (0 = No, 1 = Yes)

## 🚀 Project Structure

```
Loan Prediction/
│
├── Loan prediction challenge/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   ├── sample_submission.csv        # Sample submission format
│   └── submission.csv               # Generated predictions
│
├── Loan payback.ipynb               # Main analysis notebook
└── README.md                        # This file
```

## 🔧 Installation & Requirements

### Required Libraries

```python
# Data manipulation
pandas
numpy

# Visualization
matplotlib
seaborn
plotly

# Machine Learning
scikit-learn
xgboost
lightgbm
catboost

# Utilities
warnings
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost
```

## 📓 Notebook Contents

### 1. Data Loading and Initial Exploration
- Load training and test datasets
- Display basic statistics and data types
- Check for missing values
- Analyze target variable distribution

### 2. Exploratory Data Analysis (EDA)
- **Numerical Features Analysis**
  - Distribution plots for all numerical variables
  - Box plots to detect outliers
  - Comparison by loan payback status
  
- **Correlation Analysis**
  - Heatmap of feature correlations
  - Target variable correlation analysis
  
- **Categorical Features Analysis**
  - Distribution of categorical variables
  - Payback rate by categories
  - Detailed statistics for each category

### 3. Feature Engineering
Created **7 new features** to improve model performance:

1. **income_to_loan_ratio**: Financial capacity indicator
2. **credit_category**: Credit score bins (Poor/Fair/Good/Very Good/Excellent)
3. **debt_burden**: Total debt amount
4. **grade**: Extracted from grade_subgrade
5. **high_risk**: Binary flag for high-risk profiles
6. **loan_to_income_pct**: Loan burden percentage
7. **interest_category**: Interest rate categories (Low/Medium/High/Very High)

### 4. Model Development and Comparison

Trained and compared **6 machine learning models**:

1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble tree-based model
3. **Gradient Boosting**: Sequential boosting algorithm
4. **XGBoost**: Extreme gradient boosting
5. **LightGBM**: Light gradient boosting machine
6. **CatBoost**: Categorical boosting

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 5. Interactive Dashboards for Stakeholders

#### 📊 Dashboard 1: Executive Summary
- Loan status distribution
- Loan amount distribution
- Credit score distribution
- Interest rate distribution

#### 🎯 Dashboard 2: Risk Analysis
- Payback rate by credit score ranges
- Payback rate by employment status
- Payback rate by loan purpose
- Payback rate by education level

#### 💰 Dashboard 3: Financial Insights
- Average loan amount by grade
- Interest rate vs credit score scatter plot
- Debt-to-income ratio by payback status
- Annual income distribution

#### 👥 Dashboard 4: Demographic Analysis
- Payback rate by gender
- Payback rate by marital status
- Loan distribution by demographics

#### 🤖 Dashboard 5: Model Performance
- Model comparison chart
- Confusion matrix for best model
- ROC curves
- Feature importance analysis

### 6. Predictions & Submission
- Generate predictions for test set
- Create submission file in Kaggle format
- Display prediction distribution

### 7. Key Insights & Recommendations
- Data-driven insights for stakeholders
- Actionable recommendations for loan approval strategies

## 📈 Key Findings

### Top Predictive Features
Based on feature importance analysis from the best-performing model:
1. Credit score
2. Interest rate
3. Debt-to-income ratio
4. Loan amount
5. Annual income

### Risk Factors Identified
- **Low credit score** (< 650): Higher default risk
- **High debt-to-income ratio** (> 30%): Lower payback probability
- **Employment status**: Significant impact on loan payback
- **Loan purpose**: Varies significantly by category

## 🎓 How to Use

### Running the Complete Analysis

1. **Clone or download the project**
   ```bash
   cd "Loan Prediction"
   ```

2. **Ensure data files are in the correct location**
   - Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `Loan prediction challenge/` folder

3. **Open Jupyter Notebook**
   ```bash
   jupyter notebook "Loan payback.ipynb"
   ```

4. **Run all cells sequentially**
   - Click `Cell` → `Run All`
   - Or execute cells one by one to explore each section

### Expected Runtime
- Full analysis: ~15-30 minutes (depending on hardware)
- Data loading: ~2-5 minutes (large dataset)
- Model training: ~10-20 minutes (6 models)
- Dashboard generation: ~2-3 minutes

## 📊 Results

The notebook will generate:
- ✅ Comprehensive EDA visualizations
- ✅ 5 interactive dashboards
- ✅ Model comparison metrics
- ✅ Feature importance analysis
- ✅ Test predictions saved to `submission.csv`

## 🔍 Model Performance

Models are evaluated using stratified 80-20 train-validation split with the following metrics:
- **ROC-AUC Score**: Primary metric for model selection
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction reliability
- **Recall**: Positive case detection rate
- **F1-Score**: Harmonic mean of precision and recall

The best model is automatically selected based on ROC-AUC score.

## 💡 Business Recommendations

Based on the analysis, the following recommendations are provided:

1. **Credit Score Thresholds**: Implement stricter requirements for scores < 650
2. **DTI Ratio Limits**: Set maximum threshold at 30% for new approvals
3. **Employment Verification**: Prioritize employed applicants
4. **Loan Purpose Screening**: Carefully evaluate high-risk loan purposes
5. **Automated Decision System**: Deploy predictive model with human oversight

## 📝 Submission Format

The generated `submission.csv` file follows the Kaggle competition format:

```csv
id,loan_paid_back
593994,0
593995,1
593996,0
...
```

## 🛠️ Future Improvements

Potential enhancements for better performance:
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Advanced feature engineering (polynomial features, interactions)
- [ ] Ensemble methods combining multiple models
- [ ] Deep learning approaches (Neural Networks)
- [ ] Handle class imbalance with SMOTE or other techniques
- [ ] Cross-validation for more robust evaluation

## 👨‍💻 Author

**PhD Research Project 2025-2026**

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- The open-source community for the amazing ML libraries

## 📧 Contact

For questions or collaborations, please reach out through the project repository.

---

**Last Updated**: November 24, 2025

⭐ If you find this project helpful, please consider giving it a star!
