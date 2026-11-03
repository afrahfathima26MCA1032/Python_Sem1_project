# Loan Risk Assistant: Predicting and Explaining Small Business Loan Default — Project Status


---

## What this project is

Predicting **default risk on SBA-guaranteed small business loans**, and explaining which
application characteristics drive that risk.



## Data

**U.S. SBA National Loan Dataset** (Kaggle) — 899,164 records, 27 features, loans approved
1966–2014. Default rate 17.56%.

| Stage | Rows |
|---|---|
| Raw | 899,164 |
| After removing unlabelled | 897,167 |
| Model features | 24 |

## Results (after the `Term` leak fix)

| Metric | Value | Before the fix |
|---|---|---|
| ROC-AUC (Random Forest) | **0.842** (0.837 on the explainable model) | 0.976 |
| Default recall | **68%** (73% on the explainable model) | 89% |
| Logistic Regression baseline | 0.738 | 0.845 |
| 5-fold CV | **0.836 ± 0.002** | quoted 0.9715 — never produced by any notebook |
| Out-of-time ROC-AUC (train ≤2006, test 2007+) | **0.775** | quoted 0.946 — never produced by any notebook |
| Calibration error (ECE) | **16.9 → 0.2 points** | 7.0 → 0.2 |

**Risk tiers, validated on 179,434 held-out loans:**

| Tier | Share | Predicted | Actual |
|---|---|---|---|
| LOW (<10%) | 50.0% | 4.3% | 4.3% |
| MEDIUM (10–35%) | 34.0% | 19.0% | 18.9% |
| HIGH (>35%) | 16.0% | 56.4% | 56.1% |

**Probability accuracy:** every calibration band within 2 points of reality; below 80%,
within 1 point.

## Key findings

**1. `Term` was leaky — the biggest finding.** Repaid loans almost always carry a standard
term (84% have 60/84/120/240… months); defaulted loans almost never do (9.6%). Exactly
84-month loans default 0.6%, 83-month loans ~56%. The recorded term must have been changed
after the outcome (probably restructuring; it is *not* the loan's actual life, r = 0.05).
It inflated ROC-AUC from 0.84 to 0.98. Proved four ways in `notebooks/3_term_leak.ipynb`;
removed from every model.

**2. Three more leaky columns removed.** `ChgOffDate`, `ChgOffPrinGr` and `BalanceGross`
are recorded after the outcome is known.

**3. Imbalance handling inflates probabilities.** `class_weight="balanced"` catches more
defaults but made the model over-state risk — claiming 55% where the truth was 26%.
Isotonic calibration cut average error from 16.9 to 0.2 points without changing the ranking.

**4. Approval year is now the top feature (SHAP 22%).** Loans approved in 2005–2007 failed
heavily in the 2008 crash, so the model partly learns the economic era. The dashboard fixes
the year at 2004; the out-of-time drop to 0.775 is the same effect. A limitation to declare.

**Withdrawn:** "loan term dominates" and "borrow for longer, not less" both came from the
leaky `Term`.

## What was built

```
pythonproject/
├── src/                                    the pipeline, each step written once
│   ├── prep.py                             load, clean, features, encoding (training AND app)
│   ├── train.py                            baseline · calibration · app model → models/, outputs/
│   └── stats.py                            regional + survival figures the app reads
├── notebooks/                              analysis and charts only, import src/
│   ├── 1_data_analysis.ipynb               cleaning story, class balance, EDA, heatmap
│   ├── 2_model_results.ipynb               LR vs RF, calibration, what the model relies on (SHAP), risk tiers
│   ├── 3_term_leak.ipynb                   Term leak: evidence, impact, CV, out-of-time
│   └── 4_survival.ipynb                    Kaplan–Meier, hazard, Cox
├── app_simple.py                           Loan Risk Assistant 
├── models/model.joblib                     the app's model (18 MB, rebuilt by src/train.py)
├── data/raw/SBAnational.csv                171 MB, from Kaggle
├── data/processed/sba_clean.csv            897,167 rows, rebuilt by src/prep.py
├── data/assessments.csv                    assessments saved in the app 
├── outputs/                                charts and JSON results

```





Full rebuild, in order:

```bash
python3 -m src.train      # models + run, calibration and model results
python3 -m src.stats      # regional + survival figures for the app
```


Run the app:

```bash
streamlit run app_simple.py
```

## Methodology status — all 8 stages complete

| Stage | Status |
|---|---|
| 1. Data acquisition | ✅ |
| 2. Preprocessing | ✅ (outlier capping not performed) |
| 3. Feature engineering | ✅ |
| 4. Imbalance handling | ✅ class weighting; SMOTE deliberately rejected |
| 5. Model training | ✅ Logistic Regression + Random Forest |
| 6. Evaluation | ✅ Precision / Recall / F1 / ROC-AUC / PR-AUC, 5-fold CV, out-of-time |
| 7. Explainability | ✅ SHAP |
| 8. Risk output + contributing factors | ✅ |
| Extra: survival analysis | ✅ notebook 4 |

## Documented decisions

- **`Term` excluded from all models** — leaky (notebook 3); the dashboard no longer asks for it
- `RevLineCr == "0"` treated as `"N"` — affects 257,602 rows (29%); discarding them was not defensible
- `UrbanRural == 0` kept as its own `Undefined` level — 36% of rows
- Target encoding fitted on training data only, with smoothing for small categories
- **SMOTE rejected**: it fabricates rows by interpolating between real ones, which is
  meaningless for categorical features, and it distorts probabilities — unacceptable
- Tree depth constrained for the explainable model: ~8× faster SHAP for 0.005 ROC-AUC
- Displayed probabilities clipped to 0.5–99.5% — no credit model should claim certainty about
  an individual loan
- Dashboard fixes approval year at 2004 so the answer reflects the loan, not the economic year (2004 default rate 18.0% ≈ overall 17.6%)
- Dashboard form: franchise question removed (0.4% of SHAP; fixed to "not a franchise");
 
- Dashboard "what's driving this" chart sums SHAP values per form question (one bar per answer)
- **Bug fixed (14 Oct 2026):** the dashboard one-hot encoded a single row with
  `pd.get_dummies(drop_first=True)`, which dropped every answer, so city/countryside, credit line
  and new/established never reached the model. Now encoded by hand; verified identical to the
  training encoding on 300 real loans

## Permanent limitations 

- **Survivorship bias** — only approved loans are present; nothing can be said about rejected
  applicants
- **Old U.S. data (approved 1966–2014)** — not validated for other economies or present-day
  conditions; weaker across a crash (out-of-time 0.775)
- **No financial-health data** — no revenue, profit, debt or credit history
- **Loan term unusable** — the only recorded term is contaminated by the outcome
- **Recent loans look artificially safe** — loans approved 2011–2014 show 1.9–7.9% default because
  many had not had time to fail when the data was collected. The model partly learns this through
  approval year; the dashboard's fixed year (2004) avoids it at prediction time, but test scores
  include these easy cases

## Rejected: adding a bankruptcy dataset

**Decision: not pursued.** The proposal deck's "bankruptcy" claim cannot be
met with any available public dataset, and the attempt was investigated and abandoned for
these reasons:

| Problem | Detail |
|---|---|
| **Wrong population** | The only US option ([American Bankruptcy Dataset](https://github.com/sowide/bankruptcy_dataset), 78,682 firm-years, 1999–2018) covers NYSE/NASDAQ **listed corporations**. SBA borrowers have a median of 4 employees and a $90,000 loan. |
| **No US SME option exists** | The SME bankruptcy datasets in the literature are Chinese (SMEsD, HAT), Polish, and Taiwanese. Choosing the right company size means choosing the wrong country. |
| **Undocumented variables** | Columns are `X1`…`X18` with no published definitions in the repository, the Kaggle page, or the survey paper — so no input form could be built for them. |
| **Redundant columns** | `X14` and `X17` are identical in every row inspected, contradicting the commonly cited variable mapping. |
| **Unexplained scaling** | Values swing by ~1000× within a column across years (e.g. `740998` → `701.854`). |
| **No join key** | No shared companies or identifiers with the SBA data — it would be two unrelated models sharing a navigation bar, not one prediction. |
| **Different user** | A bankruptcy model needs financial statements; the loan officer using this tool does not have them. |

**Conclusion.** The result would have been "US small business loan repayment" beside "US
listed-company bankruptcy" — two unrelated tools, with the title still inaccurate. The
honest fix is to narrow the title instead.

**Alternative taken instead:** survival analysis (Kaplan–Meier, Cox proportional hazards)
on the existing SBA data — done in notebook 4. It models *how long* a loan survives before
charge-off, legitimately earning the word "survival" without importing a second population.


