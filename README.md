# Workers' Compensation Claim Classification

This project predicts workers' compensation claim injury type from structured claim data for the Kaggle competition [To Grant or Not to Grant](https://www.kaggle.com/competitions/to-grant-or-not-to-grant/data). The task is an imbalanced 8-class classification problem evaluated with macro F1.

## Key Outcomes

- Built a full tabular ML workflow covering exploratory analysis, feature engineering, feature selection, model comparison, prediction, and interpretability.
- Compared CatBoost, neural networks, random forests, logistic regression, Gaussian naive Bayes, and one-vs-rest random forest baselines.
- Selected CatBoost as the best validation model with a mean validation macro F1 of **0.487**.
- Achieved a final Kaggle macro F1 score of **0.408**, ranking **2nd place** on the competition leaderboard.

## Technical Stack

| Area | Tools and skills used |
| --- | --- |
| Programming environment | Python, Jupyter |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, CatBoost, neural networks, random forests, logistic regression, naive Bayes |
| Feature engineering | missing-value imputation, domain-aware outlier handling, categorical encoding, date-derived features |
| Model evaluation | stratified validation, macro F1, class balancing |
| Interpretability | SHAP feature importance |
| Visualization | Matplotlib, Seaborn |

## Data and Target

The dataset contains workers' compensation claim records with demographic, accident, legal, medical, and administrative fields. The objective is to classify the final injury/claim type.

| Class | Description |
| ---: | --- |
| 0 | Non-Comparticipated |
| 1 | Canceled |
| 2 | Medical Only |
| 3 | Temporary |
| 4 | Permanent Partial Disability (PPD) |
| 5 | Permanent Partial Disability Non-Schedule Loss (PPDNSL) |
| 6 | Permanent Total Disability (PTD) |
| 7 | Death |

## Main Challenges

| Challenge | Design choice |
| --- | --- |
| Imbalanced multiclass prediction | Used macro F1 as the main metric, compared class-aware models, and selected CatBoost with `auto_class_weights=SqrtBalanced`. |
| Mixed feature types in claim records | Engineered temporal gaps from date fields, binary missing-form indicators, categorical encodings, and domain-specific handling for numeric outliers. |
| Explaining high-impact predictions | Used SHAP values to inspect which claim attributes most influenced the final classifier. |

## Modeling Workflow

The workflow was organized into five notebooks:

1. Exploratory analysis and preprocessing
2. Feature selection
3. Model selection and hyperparameter tuning
4. Prediction on the Kaggle test set
5. Feature importance analysis

Preprocessing included missing-value treatment, domain-specific outlier handling, date feature engineering, and categorical encoding. Two complementary encodings were used for non-target categorical variables:

- **Frequency encoding** to represent category prevalence.
- **Target ordinal encoding** to order categories by their relationship with the target.

Feature selection combined correlation-based filtering with recursive feature elimination:

<p align="center">
  <img src="https://github.com/user-attachments/assets/167f98f8-ddab-46b4-bec3-cde2278412e4" alt="Feature selection workflow" width="90%">
</p>

*Note: the workflow image contains a typo from the original report. The train split is 2/3, not 1/3.*

## Results

| Model | Parameters | Train macro F1 | Mean validation macro F1 |
| --- | --- | ---: | ---: |
| **CatBoost** | `depth=6`, `auto_class_weights=SqrtBalanced`, `loss_function=MultiClassOneVsAll` | **0.560** | **0.487** |
| Neural Network | `hidden_layer_sizes=(25, 8)`, `learning_rate_init=0.01` | 0.452 | 0.447 |
| Random Forest | `max_depth=6`, `class_weight=balanced` | 0.429 | 0.405 |
| Logistic Regression | `C=1`, `solver=lbfgs`, `class_weight=None` | 0.426 | 0.395 |
| Gaussian Naive Bayes | `var_smoothing=0.1` | 0.351 | 0.317 |
| OVR Random Forest | `max_depth=6`, `class_weight=balanced` | 0.332 | 0.314 |

CatBoost had the best validation performance and was used for the final Kaggle submission.

| Submission | Macro F1 | Result |
| --- | ---: | --- |
| Final Kaggle submission | **0.408** | **2nd place** on the [leaderboard](https://www.kaggle.com/competitions/to-grant-or-not-to-grant/leaderboard) |

## Model Interpretation

SHAP analysis highlighted the variables with the largest influence on the CatBoost classifier:

- **Wage**: higher wages were generally associated with higher claim values.
- **Days to First Hearing**: shorter time to the first hearing often correlated with more severe claims.
- **Attorney Involvement**: attorney involvement tended to appear in higher-severity claims.
- **IME-3 Count**: more IME-3 evaluations were associated with increased severity.
- **Missing C-3 Form**: missing C-3 forms were linked to more severe outcomes, including fatal claims.
- **Part of Body Injured**: some injury locations were more strongly associated with severe or fatal claims.
- **Accident-Assembly Gap (Days)**: longer gaps were associated with canceled claims.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b48d47d3-0b28-4195-845d-92ae26ba21d4" alt="SHAP feature importance summary" width="85%">
</p>

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── utils.py
├── data/
│   └── geo-data.csv
└── notebooks/
    ├── 00_eda_preprocessing.ipynb
    ├── 01_feature_selection.ipynb
    ├── 02_model_selection_and_tuning.ipynb
    ├── 03_prediction_on_test.ipynb
    └── 04_feature_importances.ipynb
```

## Reproducing the Workflow

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle competition files and place them in `data/`:

```text
data/
├── train_data.csv
├── test_data.csv
└── geo-data.csv
```

Then run the notebooks in numerical order. The first notebook creates the engineered training file used by the later notebooks:

```text
data/train_new_feats.csv
```
