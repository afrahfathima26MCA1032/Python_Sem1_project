# Loan Risk Assistant
### Predicting and Explaining Small Business Loan Default

A data-analysis and machine-learning project on the U.S. Small Business Administration (SBA)
loan dataset. Given the details of a new small-business loan, the system estimates the
**probability that the loan will not be paid back**, explains **why**, and suggests **when
to review it**, through an interactive Streamlit web application.

| | |
|---|---|
| **Dataset** | SBA National loans, 1966–2014 — 899,164 loans, 27 columns |
| **Task** | Binary classification: charged off (1) vs paid in full (0) |
| **Model** | Random Forest with isotonic probability calibration |
| **Explainability** | SHAP (per-loan reasons) |
| **Extra analysis** | Survival analysis (Kaplan–Meier, Nelson–Aalen, Cox) |
| **Interface** | Streamlit web app with Plotly charts |
| **Result** | ROC-AUC **0.842** (baseline Logistic Regression 0.738), calibration error **0.2 points** |

---

## Methodology

### 1. Problem definition
The target is built from the `MIS_Status` column: `CHGOFF` (loan charged off, i.e. not repaid)
becomes **1**, `P I F` (paid in full) becomes **0**. The model may use only information that is
known **on the day the loan is approved**, so that it can be used for real decisions.

### 2. Data source
The *SBA National* dataset (Li, Mickel & Taylor, 2018), the base paper of this project:
899,164 loans and 27 columns describing the business, the loan, the bank and the outcome.

### 3. Data cleaning — `src/prep.py`
1. **Removed leaky columns** that are only filled in after a loan fails:
   `ChgOffDate`, `ChgOffPrinGr`, `BalanceGross`.
2. **Removed identifier columns** that cannot be learned from: loan number, name, city, zip, bank name.
3. Dropped loans with no recorded outcome → **897,167 loans** remain; **17.6 %** defaulted.
4. Converted money columns (`"$60,000.00"`) to numbers and cleaned the approval year (`"1976A"` → 1976).
5. Standardised category codes (`RevLineCr`, `LowDoc`, `NewExist`, `UrbanRural`); invalid codes become `Unknown`.
6. Mapped the 6-digit NAICS code to **20 industry sectors** and the franchise code to a 0/1 flag.

### 4. Feature engineering
Five new features were created:

| Feature | Meaning |
|---|---|
| `guarantee_ratio` | Share of the loan guaranteed by the SBA |
| `disbursement_ratio` | Share of the approved amount actually paid out |
| `loan_per_employee` | Loan size relative to business size |
| `same_state` | 1 if the bank is in the same state as the business |
| `jobs_total` | Jobs created plus jobs retained |

### 5. Exploratory data analysis — `notebooks/1_data_analysis.ipynb`
Class balance, default rate by industry and by state, an industry × state heatmap,
a correlation matrix, distributions of repaid vs defaulted loans and an outlier analysis.
Key findings:
- The data is **imbalanced**: 82 % repaid, 18 % defaulted.
- Default rates range from about 8 % (Mining, Agriculture) to 29 % (Real Estate), and from
  7 % (Montana) to 27 % (Florida).
- Loans approved in 2005–2008 defaulted far more often (the 2008 financial crisis).
- No single numeric column correlates strongly with default (|r| ≤ 0.20), so a multivariate model is needed.
- About 10 % of values are statistical outliers; they are genuine loans and were kept.

### 6. Leakage investigation — `notebooks/3_term_leak.ipynb`
A first model reached ROC-AUC **0.976**, which is unrealistically high for credit risk.
Each column was then tested on its own: the loan **`Term`** alone scored 0.81, against 0.69
for the next-best column. 84 % of repaid loans have a standard term, against only 9.6 % of
defaulted ones, and the recorded term does not match how long the loan actually lasted
(r = 0.05). The recorded term therefore carries information from after the outcome
(most likely loan restructuring), so **`Term` was removed from every model**.
The honest ROC-AUC is **0.842**.

### 7. Encoding — `src/prep.py`
- **Smoothed target encoding** for `State`, `BankState` and industry: each value is replaced
  by its historical default rate, pulled towards the overall rate for small groups
  (smoothing = 100). The rates are learned from the **training set only** to avoid leakage.
- **One-hot encoding** for the yes/no style category columns.
- Missing values are filled with the training-set median.
- The final model input has **24 numeric features**.

### 8. Model training — `src/train.py`
- **Data split:** 60 % training, 20 % calibration, 20 % test, stratified to keep the 82/18
  ratio, with a fixed random seed (42) for reproducibility.
- **Baseline:** Logistic Regression (standardised inputs).
- **Main model:** Random Forest, 100 trees, `min_samples_leaf = 50`, `max_depth = 20`.
- **Class imbalance:** handled with `class_weight="balanced"` (each default weighted about 4.7×)
  instead of synthetic oversampling.

### 9. Probability calibration
Class weighting pushes the forest's raw scores too high (average 39 % against a real 17.6 %).
An **isotonic calibration** layer was fitted on the separate 20 % calibration set, with the
forest frozen. Isotonic was chosen over sigmoid because it gave the lower error.
Expected Calibration Error dropped from **16.9 to 0.2 percentage points**, with ROC-AUC unchanged.

### 10. Evaluation — `notebooks/2_model_results.ipynb`

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.738 | 0.366 |
| **Random Forest** | **0.842** | **0.589** |

- 5-fold cross-validation: ROC-AUC **0.836 ± 0.002** (stable).
- Out-of-time test (trained on earlier years, tested on later years): 0.775.
- App model on the test set: recall 73 %, calibration error 0.2 points.
- Predicted vs actual default rate per risk tier: Low ≈ 4 %, Medium ≈ 19 %, High ≈ 56 %.

### 11. Explainability
**SHAP** (TreeExplainer) computes how much each feature pushed a loan's risk up or down.
In the app, SHAP values are summed per form question, so the user sees plain reasons
such as *"Bank in another state — raises risk"*.

### 12. Survival analysis — `notebooks/4_survival.ipynb`
Kaplan–Meier survival curves, the Nelson–Aalen hazard rate and a Cox proportional-hazards
model estimate **when** loans fail. Overall, failures peak around **month 43**. Each industry's
peak month is used by the app to suggest a review date six months earlier.

### 13. Application — `app_simple.py`
A Streamlit web app with five pages:

| Page | What it does |
|---|---|
| **New Assessment** | Enter a loan → calibrated risk ("X in 100"), risk level, top reasons (SHAP), suggested next step, review date |
| **My Assessments** | Saved assessments with summary figures; export to CSV |
| **Data Analysis** | Key charts from the exploratory analysis |
| **Market Lookup** | Interactive map and charts of historical default rates by state and industry |
| **About the Model** | Model performance and calibration check |

Risk levels: **Lower** (< 10 %), **Medium** (10–35 %), **Higher** (≥ 35 %).

---

## Project structure

```
├── app_simple.py                    Streamlit web application (Loan Risk Assistant)
├── PROJECT_CONTEXT.md               project status, results, decisions and limitations
├── requirements.txt
├── src/
│   ├── prep.py                      data cleaning, feature engineering, encoding
│   ├── train.py                     baseline, calibration and final model training
│   └── stats.py                     regional default rates and survival statistics for the app
├── notebooks/
│   ├── 1_data_analysis.ipynb
│   ├── 2_model_results.ipynb
│   ├── 3_term_leak.ipynb
│   └── 4_survival.ipynb
├── data/
│   ├── raw/SBAnational.csv          original SBA dataset (899,164 loans)
│   ├── processed/sba_clean.csv      cleaned dataset (897,167 loans), made by src/prep.py
│   └── assessments.csv              example assessments saved from the app
├── models/
│   ├── model.joblib                 trained model, calibrator and encoder used by the app
│   └── rf_explainable.joblib        earlier version of the explainable model
├── outputs/                         result summaries (JSON) and charts (figures/)
└── .streamlit/config.toml           app theme
```

The two dataset files are larger than GitHub's 100 MB file limit, so they are attached to the
**v1.0 Release** of this repository. Download SBAnational.csv into data/raw/ and
sba_clean.csv into data/processed/.

## How to run

```bash
pip install -r requirements.txt
streamlit run app_simple.py
```

To reproduce every result from the dataset:

```bash
python -m src.train      # trains the models  -> models/, outputs/
python -m src.stats      # regional and survival statistics -> outputs/
```

The notebooks can then be run in order (1 → 4) to regenerate the analysis and charts.

## Limitations
- The model knows nothing about the business's finances (revenue, profit, debt).
- Loans approved in 2011–2014 look safer than they are, because many had not yet had time to fail.
- Performance is lower on later years (out-of-time ROC-AUC 0.775), as lending conditions change over time.
- The risk-level thresholds and suggested next steps are business rules, not learned by the model.

## Reference
Li, M., Mickel, A., & Taylor, S. (2018). *"Should This Loan be Approved or Denied?": A Large
Dataset with Class Assignment Guidelines.* Journal of Statistics Education, 26(1).
