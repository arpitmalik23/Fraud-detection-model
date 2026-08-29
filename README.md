# Credit Default Risk Prediction
Machine Learning pipeline to classify bank customers as **defaulters vs non-defaulters** using real credit account data with high class imbalance.

---

## Problem Statement  
Financial institutions face increasing risk due to delayed payments and loan defaults.  
The goal of this project is to **predict whether a customer is likely to default** based on historical credit & repayment behavior.

✅ Binary Classification (0 = Non-defaulter, 1 = Defaulter)  
✅ Highly imbalanced data (approx. 89% non-defaulters, 11% defaulters)  
✅ Business objective: **maximize F1 score & AUPRC for positive class (defaulters)**

---

## Dataset Overview  
| Description | Value |
|-------------|--------|
| Total rows | ~310K |
| Total features | 139 raw features |
| Target variable | `DEFAULT_FLAG` |
| Class ratio | 1 : 7.7 (defaulters vs non-defaulters) |
| Data type mix | Numeric + categorical + temporal |

---

## End-to-End Pipeline
Data Loading → EDA → Feature Engineering → Train-Test Split →
Resampling + Class Weighting → Model Training → Threshold Tuning → Explainability (SHAP)


---

## Feature Engineering Summary  
✔ Aggregated transactional behavior (rolling means, sums, differences)  
✔ Utilization ratios & credit limits  
✔ Statistical summary features (mean, std, min, max per metric)  
✔ Derived delay counts / repayment gaps  
✔ Removed multicollinear features using correlation heatmap  
✔ Standardization applied to numerical features

---

## Handling Class Imbalance  
| Technique | Used? | Notes |
|-----------|-------|-------|
| Class weights | ✅ | `scale_pos_weight` in XGBoost |
| SMOTETomek | ✅ | Tested but kept only class weights in final model |
| Random undersampling | ❌ | Led to information loss |
| Cost-sensitive loss | ✅ | Implicit via `scale_pos_weight` |

---

## Models Trained & Compared

| Model | F1 (Class 1) | AUPRC | Notes |
|-------|--------------|-------|-------|
| Decision Tree | ~0.28 | 0.32 | Underfits |
| Random Forest | ~0.38 | 0.40 | Better but slow |
| LightGBM | ~0.40 | 0.44 | Good performance |
| **XGBoost (baseline)** | **0.44** | **0.45** | Best without hyperparameter tuning |
| **XGBoost (GridSearch tuned)** | **0.42–0.45** | **0.46** | Slight AUPRC gain, similar F1 |

✅ **XGBoost consistently outperformed others and was selected as final model**  
✅ Threshold tuning improved results more than hyperparameter optimization

---

## Final Model Performance

### ✅ Baseline XGBoost + Threshold Tuning (threshold = 0.6)
- F1 (defaulters): 0.4408
- AUPRC: 0.4541
- Recall: 0.6056
- Precision: 0.3465

### ✅ XGBoost + GridSearchCV
- F1 (defaulters): 0.42 – 0.45
- AUPRC: 0.4588

🔹 GridSearchCV improved **AUPRC**,  
🔹 Baseline model with tuned threshold gave slightly better **F1**.

> 📌 Insight: In imbalanced classification, **threshold tuning > hyperparameter tuning**  
> for improving F1 score on minority class.

---

## Threshold Tuning  
Used `precision_recall_curve` sweep to maximize F1:

```python
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
```
Default 0.5 → poor F1
Best threshold ≈ 0.6 → +15% improvement in F1

---

## Model Explainability (SHAP)
```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```
- Global feature importance
- Local explanations for single customers
- Identified top drivers of default risk

---

## Conclusion

- XGBoost proved to be the best model for credit default prediction
- F1 & AUPRC improved through class weights + threshold tuning
- GridSearch improved AUPRC slightly, but baseline tuned model remained competitive
- SHAP enabled full explainability for business stakeholders
