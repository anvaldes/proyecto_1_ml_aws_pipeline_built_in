# 🧠 End-to-End XGBoost ML Pipeline on AWS SageMaker

This repository contains a full machine learning pipeline for **training, hyperparameter tuning, evaluation**, and **model registration** using **SageMaker Pipelines**.  
It is designed to process tabular data stored in S3 and train an **XGBoost classifier**, with a focus on modularity and automation.

---

## 🚀 Features

- ✅ Automated pipeline using SageMaker's `Pipeline` SDK  
- 🔄 Preprocessing step using built-in `scikit-learn` image  
- 🔍 Hyperparameter Tuning using `HyperparameterTuner`  
- 💾 Model saving in JSON format for portability  
- 📈 Evaluation using `f1_score` and `classification_report`  
- 📝 Model registration for production use  
- ☁️ Full integration with Amazon S3

---

## ☁️ SageMaker Pipeline Flow

### ✅ Step 1: Preprocessing

- Script: `preprocessing.py`  
- Input: S3 path: `s3://proyecto-1-ml/datasets/2025_06`  
- Output: `s3://proyecto-1-ml/preprocessing/2025_06/`

The script modifies `"person_home_ownership"` and copies labels.

---

### 🔍 Step 2: Hyperparameter Tuning

- Script: `train_hpt_job.py`  
- Objective metric: `f1_score`  
- Search space:
  - `n_estimators`: 2 to 10
  - `max_depth`: 2 to 10

Saves model in `/opt/ml/model/model.json`

---

### 💾 Step 3: Model Registration

- The top model is uploaded to SageMaker Model Registry using `ModelStep`.

---

### 📊 Step 4: Evaluation

- Script: `evaluate.py`  
- Reads `model.tar.gz` and extracts `model.json`  
- Runs prediction on test set and logs:
  - `f1_score` (train/val/test)
  - `classification_report`
- Saves report to:

```bash
s3://proyecto-1-ml/evaluation/2025_06/report.json
```

---
