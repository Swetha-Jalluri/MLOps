# MLOps Lab 1 – Wine Quality Prediction

## Objective

The objective of this lab is to apply MLOps concepts to a simple end-to-end machine learning workflow.  
Instead of implementing basic arithmetic operations, this lab focuses on building and managing a **machine learning model** that predicts wine quality.  
The workflow demonstrates data processing, model training, testing, CI/CD, experiment tracking, and deployment.

---

## Learning Outcomes

After completing this lab, the following outcomes are achieved:

1. Structured ML project organization  
2. Data preprocessing and feature scaling  
3. Model training and hyperparameter tuning  
4. Automated testing using `pytest` and `unittest`  
5. Continuous Integration with GitHub Actions  
6. Experiment tracking using MLflow  
7. Model deployment through a Flask REST API  

---

## Lab Description

### Problem Statement
Predict whether a wine sample is **Good** (quality ≥ 7) or **Regular** (quality < 7) using 11 physicochemical features from the **UCI Wine Quality Dataset**.

### Tasks Performed
1. Load and preprocess the dataset  
2. Train a **Random Forest Classifier**  
3. Evaluate model performance (accuracy, precision, recall, F1-score)  
4. Track experiments using **MLflow**  
5. Deploy model through a **Flask API**  
6. Validate functionality using automated tests  

---

## Lab Structure

```
Lab1_New/
│
├── src/
│   ├── wine_model.py
│   ├── app.py
│   └── __init__.py
│
├── tests/
│   ├── test_wine_model_pytest.py
│   ├── test_unittest_basic.py
│   └── test_app_api.py
│
├── artifacts/              
│   ├── model.pkl            # Trained model file
│   ├── scaler.pkl           # StandardScaler used during preprocessing
│   ├── metrics.json         # Saved model metrics (accuracy, F1, etc.)
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── mlruns/                  
│   └── ...                  
│
├── .github/
│   ├── pytest_action.yml
│   └── unittest_action.yml
│
├── requirements.txt
├── .gitignore
├── README.md
└── venv/                    # (not tracked in git)


## Environment Setup

### Step 1: Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # For Windows
# or
source venv/bin/activate   # For macOS/Linux
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Lab

### Step 1: Train and Evaluate the Model
```bash
python src/wine_model.py
```

**Expected Output:**
```
Training Wine Quality Model (with tuning)...

Model Performance:
accuracy: 0.9375
precision: 0.9259
recall: 0.5814
f1_score: 0.7143
```

The trained model (`model.pkl`) and scaler (`scaler.pkl`) will be saved automatically.

---

## Experiment Tracking (MLflow)

To view experiment results:
```bash
mlflow ui
```

Then open in your browser:
```
http://127.0.0.1:5000
```

**MLflow logs include:**
- Model parameters (e.g., `n_estimators`, `max_depth`)  
- Metrics (accuracy, precision, recall, F1)  
- Artifacts:
  - Feature importance chart  
  - Confusion matrix  
  - ROC curve  
  - Trained model and scaler  

---

## API Deployment

The trained model can be accessed through a REST API.

### Start the Flask API
```bash
python src/app.py
```

The API runs at:
```
http://127.0.0.1:5001
```

#### Health Check Endpoint
```
GET /health
Response: {"status": "ok"}
```

#### Prediction Endpoint
```
POST /predict
{
  "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]
}
```

Response:
```
{
  "prediction": 0,
  "label": "Regular"
}
```

---

## Testing

### Run Pytest
```bash
pytest -q
```

**Expected Output:**
```
............                                                                               [100%]
12 passed, 1 warning in 3.9s
```

### Run Unittest
```bash
python -m unittest discover tests -v
```

Tests verify:
- Data loading and preprocessing  
- Model training and predictions  
- Metric calculations  
- API functionality  

---

## Continuous Integration (GitHub Actions)

Two CI workflows are configured in the `.github/workflows` directory.

1. **pytest_action.yml**  
   - Executes all pytest test cases on push or pull requests.  
2. **unittest_action.yml**  
   - Executes unittest cases independently.  

Each workflow performs:
- Python environment setup  
- Dependency installation  
- Automated testing  
- Report generation  

CI results can be viewed under the **Actions** tab in the GitHub repository.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | 93.75% |
| Precision | 92.59% |
| Recall | 58.14% |
| F1-Score | 71.43% |

### Top 5 Important Features
1. Alcohol  
2. Sulphates  
3. Density  
4. Volatile Acidity  
5. Citric Acid  

---

## Future Work
- Extend API for batch prediction using CSV input  
- Containerize application using Docker  
- Integrate with a remote MLflow tracking server  
- Automate deployment through CI/CD  

---

## Conclusion

This lab demonstrates how machine learning models can be integrated into an MLOps workflow.  
It covers data preparation, model training, experiment logging, testing, and deployment automation.  
All tests have passed successfully, and the model achieves strong predictive performance on the wine quality dataset.

---

## License

This repository is part of the **MLOps Laboratory Assignments (Lab 1)**.  
All rights reserved © 2025.
