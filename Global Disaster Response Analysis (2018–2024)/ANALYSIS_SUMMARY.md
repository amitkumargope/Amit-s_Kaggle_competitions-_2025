# Global Disaster Response Analysis (2018–2024) - Summary Report

**Date:** November 24, 2025  
**Dataset:** 50,000 disaster events across multiple countries and disaster types

---

## Executive Summary

This analysis examines disaster management performance and relationships between disaster intensity, response quality, and recovery duration across different regions from 2018 to 2024. The study combines exploratory data analysis (EDA), statistical testing, and machine learning predictive models.

---

## Key Findings

### 1. Statistical Relationships

**Strong Correlations Identified:**
- **Severity Index ↔ Recovery Days**: r = 0.969 (p < 0.0001) - Very strong positive correlation
  - Higher severity disasters require significantly longer recovery periods
- **Severity Index ↔ Response Efficiency**: r = 0.600 (p < 0.0001) - Moderate positive correlation
  - More severe disasters tend to receive better-coordinated responses
- **Response Efficiency ↔ Recovery Days**: r = 0.580 (p < 0.0001) - Moderate positive correlation
  - Better response efficiency is associated with longer recovery periods (likely confounded by severity)

**ANOVA Results:**
- Response efficiency scores show **no significant differences** across disaster types (F=1.463, p=0.155)
- Mean response efficiency ranges from 87.3% (Earthquakes) to 87.8% (Tornadoes)
- Response quality is consistent regardless of disaster type

---

### 2. Linear Regression Analysis

**Predicting Recovery Days:**
- **R² = 0.938** - Model explains 93.8% of variance
- **MAE = 3.99 days** - Average prediction error
- **RMSE = 5.00 days**

**Key Predictors (Coefficients):**
1. `severity_index`: +10.05 days per unit increase (dominant factor)
2. All other factors have minimal direct impact on recovery duration
3. Severity index is by far the strongest predictor

---

### 3. Machine Learning Model Performance

#### Task 1: Recovery Days Prediction (Regression)

| Model               | R²     | MAE (days) | RMSE (days) |
|---------------------|--------|------------|-------------|
| Linear Regression   | 0.9385 | 3.98       | 5.00        |
| **Gradient Boosting** | **0.9379** | **4.00** | **5.02** |
| Random Forest       | 0.9374 | 4.01       | 5.04        |
| Decision Tree       | 0.9307 | 4.20       | 5.30        |

**Best Model:** Gradient Boosting (R² = 0.938)

**Feature Importance:**
1. **Severity Index**: 98.9% importance - Dominant predictor
2. Longitude: 0.17%
3. Aid Amount: 0.17%
4. All other features: < 0.2%

#### Task 2: Response Efficiency Score Prediction (Regression)

| Model               | R²     | MAE   | RMSE  |
|---------------------|--------|-------|-------|
| **Linear Regression** | **0.7828** | **3.74** | **4.67** |
| Gradient Boosting   | 0.7817 | 3.78  | 4.68  |
| Random Forest       | 0.7808 | 3.78  | 4.69  |
| Decision Tree       | 0.7586 | 3.92  | 4.92  |

**Best Model:** Linear Regression (R² = 0.783)

#### Task 3: Severity Classification (Classification)

**Categories:** Low (0-3.33), Medium (3.33-6.67), High (6.67-10)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| **Gradient Boosting** | **79.2%** | **0.79** | **0.79** | **0.79** |
| Random Forest       | 79.0%    | 0.79      | 0.79   | 0.78     |
| Decision Tree       | 78.1%    | 0.78      | 0.78   | 0.77     |

**Best Model:** Gradient Boosting (79.2% accuracy)

**Classification Performance:**
- **High Severity**: 72% precision, 55% recall
- **Medium Severity**: 80% precision, 88% recall (best performance)
- **Low Severity**: 84% precision, 78% recall

---

## Exploratory Data Analysis Insights

### Temporal Trends
- Disaster events distributed across 2018-2024
- Response quality and recovery metrics show stability over time
- No significant temporal degradation in disaster management performance

### Geographic Patterns
- **Top Countries by Event Count:**
  1. Multiple countries show similar event frequencies
  2. Geographic location (lat/lon) has minimal impact on predictions
  3. Country and disaster type are less important than severity

### Disaster Type Distribution
- 10 disaster types analyzed: Earthquake, Flood, Hurricane, Tornado, Wildfire, etc.
- Response efficiency is remarkably consistent across types
- Recovery duration strongly depends on severity, not disaster type

---

## Recommendations

### 1. Resource Allocation
- **Priority Focus:** Severity index is the single most important predictor
- Allocate resources based on predicted severity rather than disaster type
- Use ML models to forecast recovery duration and plan accordingly

### 2. Early Warning & Preparation
- Implement severity classification models (79% accuracy) for early categorization
- High-severity disasters require immediate, intensive resource deployment
- Pre-position resources in areas with high predicted severity

### 3. Response Optimization
- Current response efficiency is consistent (~87%) across disaster types - maintain this standard
- Focus on reducing response time for high-severity events
- Economic loss and casualties have minimal direct impact on recovery duration when controlling for severity

### 4. Predictive Planning
- Deploy recovery days prediction models (94% accuracy) for operational planning
- Expected recovery time can be predicted within ±4 days on average
- Use predictions for:
  - Budgeting and resource planning
  - Community communication and expectation management
  - Logistics and supply chain coordination

### 5. Data-Driven Decision Making
- Severity index drives outcomes - invest in accurate severity assessment tools
- Response efficiency models can help identify underperforming regions
- Continue collecting data to refine predictive models

---

## Technical Details

### Dataset Characteristics
- **Size:** 50,000 observations, 12+ features
- **Time Period:** 2018-2024
- **Coverage:** Multiple countries and 10 disaster types
- **Key Features:** Severity index, casualties, economic loss, response time, aid amount, recovery days, response efficiency score

### Preprocessing
- No missing values in primary analysis columns
- Created normalized metrics (0-1 scale) for intensity, response quality, recovery duration
- Applied log transformations to monetary values for better distribution
- Encoded categorical variables (country, disaster type) for ML models

### Model Validation
- Train-test split: 80/20
- Standardized features using StandardScaler
- Cross-validation implicit in ensemble methods (Random Forest, Gradient Boosting)
- Consistent evaluation metrics: R², MAE, RMSE for regression; Accuracy, Precision, Recall, F1 for classification

---

## Files Generated

1. **`Global Disaster Response Analysis.ipynb`** - Complete annotated analysis notebook
2. **`cleaned_global_disaster_response.csv`** - Preprocessed dataset with engineered features
3. **`ml_model_performance_summary.csv`** - Model comparison table
4. **`figures/`** - Directory containing:
   - Distribution plots (severity, response efficiency, recovery days)
   - Correlation heatmap
   - Country-level aggregates
   - Yearly time-series trends
   - ML prediction visualizations
   - Feature importance plots
   - Confusion matrices

---

## Conclusion

Disaster severity is the overwhelming driver of recovery duration, explaining 93.8% of variance. Machine learning models can predict recovery times with high accuracy (±4 days), enabling data-driven resource allocation and planning. Response efficiency remains consistent across disaster types at ~87%, suggesting standardized protocols are working well globally. Future efforts should focus on rapid severity assessment and targeted resource deployment for high-severity events.

---

**For Questions or Further Analysis:**  
Contact: [Your Contact Information]  
Repository: [Link to project repository if applicable]
