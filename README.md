# Risk Assessment and Claims Prediction

This project aims to enhance risk assessment, premium pricing, and claims prediction in the insurance industry by applying various machine learning and statistical modeling techniques. It is built to help insurers make data-driven decisions that improve profitability, reduce risk, and enhance customer satisfaction.

## Project Overview

The project is divided into four main tasks:

1. **Exploratory Data Analysis (EDA)**
2. **A/B Hypothesis Testing**
3. **Statistical Modeling**
4. **Feature Engineering & Model Tuning**

---

### Task 1: Exploratory Data Analysis (EDA)

In this phase, we explored the dataset to understand its structure, distributions, and key insights. Several steps were taken:

- **Data Cleaning**: Missing values were handled appropriately.
- **Descriptive Statistics**: Summary statistics were generated to understand the range, distribution, and variance of key features.
- **Visualizations**: Heatmaps, histograms, bar charts, and scatter plots were used to identify patterns and correlations between features.

#### Key Insights:

- Certain provinces have higher average risk levels, impacting premium pricing.
- Total claims and profit margins vary significantly across zip codes.
- Results from EDA helped in identifying trends and anomalies which laid the foundation for the hypothesis testing and modeling stages.

---

### Task 2: A/B Hypothesis Testing

A/B testing was conducted to validate several hypotheses regarding insurance risk, profit margins, and customer behavior. The null hypotheses tested were:

- No risk differences across provinces
- No risk differences between zip codes
- No significant margin (profit) difference between zip codes
- No significant risk difference between Women and Men

#### Methods:

- Categorical data were tested using chi-square tests.
- Numerical data were tested using t-tests and z-tests.

#### Results:

- The null hypothesis for differences in risk across provinces was rejected, indicating that province does influence risk levels.
- Risk levels and margins were found to vary significantly by gender and zip codes, providing valuable insights for more accurate premium calculations.

---

### Task 3: Statistical Modeling

This phase involved building predictive models to forecast claims and risks based on customer demographics and vehicle details. We experimented with three different models:

- Linear Regression
- Random Forest
- XGBoost

#### Data Preparation:

- **Missing Data**: Imputation strategies were applied to handle missing values.
- **Encoding**: Categorical features were encoded using one-hot encoding.
- **Feature Selection**: Features like `SumInsured`, `CalculatedPremiumPerTerm`, and `VehicleIntroDate` were identified as key drivers for predictions.

#### Model Performance:

| Model            | RMSE      | MAE      |
|------------------|-----------|----------|
| Linear Regression| 0.708469  | 0.075692 |
| Random Forest    | 0.942831  | 0.075886 |
| XGBoost          | 0.886449  | 0.072435 |

### Hyperparameter Tuned of random forest and xgboost
| Model            | RMSE      | MAE      |
|------------------|-----------|----------|
| Random Forest    | 0.776032  | 0.065670 |
| XGBoost          | 0.706291  | 0.063322 |

XGBoost outperformed the others with the lowest error rates (RMSE, MAE) and better overall prediction quality.

---

### Task 4: Feature Engineering & Model Tuning

In this task, we focused on refining the models and extracting insights using advanced techniques:

- **Feature Engineering**: New features like `VehicleAge` and `CapitalOutstanding` were engineered to better capture the relationships between the input variables and target variables (`TotalPremium`, `TotalClaims`).
- **SHAP Analysis**: SHAP (SHapley Additive exPlanations) was used to explain the model predictions and understand which features contributed the most to the output.

#### Key influential features:

- `SumInsured`
- `CalculatedPremiumPerTerm`
- `VehicleIntroDate`

These analyses helped validate the model and provided valuable business insights for premium pricing.

---

## Project Structure

The repository is structured as follows:

```bash
.github/
Data/
notebooks/
    ├── EDA.ipynb
    ├── HypothesisTesting.ipynb
    ├── Model.ipynb
    ├── README.md
scripts/
    ├── eda.py
    ├── hypothesisTesting.py
    ├── modeling.py
src/
tests/
.gitignore
README.md
requirements.txt
notebooks/: Contains Jupyter notebooks for each task including EDA, hypothesis testing, and modeling.
scripts/: Python scripts for conducting EDA, hypothesis testing, and modeling in a modular and reusable format.
```

- `notebooks/`: Contains Jupyter notebooks for each task including EDA, hypothesis testing, and modeling.
- `scripts/`: Python scripts for conducting EDA, hypothesis testing, and modeling in a modular and reusable format.
- `src/`: Core functions and modules for the project.
- `tests/`: Test cases to ensure code robustness.

---

## Key Learnings and Insights

- Risk varies significantly across geographic and demographic features, suggesting insurers can refine their pricing strategies based on region and customer profile.
- Feature importance analysis via SHAP provided critical insights into which variables drive premium calculations and claims.
- XGBoost outperformed other models due to its ability to capture complex relationships between features.

---

## How to Run

### Clone the repository:

```bash
git clone https://github.com/yerosan/Insurance_solutions.git
cd Insurance_solutions


```
Install Dependencies:

```bash
pip install -r requirements.txt
```
### Run the Jupyter notebooks for each task:
  - EDA: notebooks/EDA.ipynb
  - Hypothesis Testing: notebooks/HypothesisTesting.ipynb
  - Modeling: notebooks/Model.ipynb

### Execute Python scripts from the terminal for automated execution:

```bash
python scripts/eda.py
python scripts/hypothesisTesting.py
python scripts/modeling.py
```