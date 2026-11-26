# Employee Attrition with Logistic Regression

I used the IBM HR Analytics dataset to study employee attrition and compare three ways to classify departures. The dataset contains 1,470 employee records, including 237 departures (16.1%). I focused on how threshold adjustment and class weighting change the balance between catching departures and raising false positives.

## Approach

1. Examined the class distribution, compared age and monthly income across attrition groups, and used a correlation heatmap to identify redundant features.
2. Removed the employee identifier, constant columns, and selected correlated features. One-hot encoded categorical variables and used an 80/20 stratified train/test split.
3. Fit `StandardScaler` on the training set, then trained a baseline logistic regression and a second model with `class_weight="balanced"`. I also evaluated the baseline with a lower probability threshold.

## Results

| Model and decision threshold | Accuracy | Attrition precision | Attrition recall | Attrition F1 | Departures detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (0.50) | 86% | 62% | 32% | 0.42 | 15 / 47 |
| Baseline (0.30) | 84% | 51% | 49% | 0.50 | 23 / 47 |
| Class-weighted model (0.50) | 78% | 39% | 66% | 0.49 | 31 / 47 |

The baseline achieved a **ROC AUC of 0.81** on the 294-record test set. Class weighting detected 16 more departures than the baseline, while false positives increased from 9 to 48. The lower-threshold baseline detected 23 departures with 22 false positives.

Overtime and frequent business travel had positive coefficients in the baseline model, while total working years and job satisfaction had negative coefficients. These are associations within the fitted model. The comparison uses a single test split, so any operating threshold would need separate validation.

## Run the notebook

Download the IBM HR Analytics Employee Attrition dataset separately and place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the repository root. The CSV is excluded from version control.

```bash
python -m pip install -r requirements.txt
jupyter notebook "IBM HR Attrition.ipynb.ipynb"
```

The [original notebook](IBM%20HR%20Attrition.ipynb.ipynb) includes the saved charts, confusion matrices, and classification reports.
