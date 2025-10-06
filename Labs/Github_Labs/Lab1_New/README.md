# 🍷 Wine Quality Prediction Model - MLOps Lab 1

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [My Contributions](#my-contributions)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Model Details](#model-details)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Results](#results)

---

## 🎯 Project Overview

This project implements a complete MLOps pipeline for predicting wine quality using machine learning. It demonstrates industry best practices including automated testing, continuous integration/continuous deployment (CI/CD) with GitHub Actions, and proper project structuring. The project replaces the original calculator example with a real-world machine learning application.

### Key Features
- **Binary Classification**: Predicts whether wine is "Good" (quality ≥ 7) or "Regular" (quality < 7)
- **Automated Testing**: Comprehensive test coverage using both pytest and unittest
- **CI/CD Pipeline**: Automated testing on every push using GitHub Actions
- **Model Persistence**: Save and load trained models for deployment
- **Feature Importance Analysis**: Identifies which wine characteristics most influence quality

---

## 🚀 My Contributions

This implementation showcases creativity and practical ML application by:

| Original Lab | My Implementation |
|-------------|-------------------|
| Simple calculator functions | Random Forest ML model for wine quality |
| Basic arithmetic operations | Real-world dataset from UCI repository |
| Mock data testing | Actual data preprocessing and validation |
| Simple function tests | ML-specific tests (accuracy, metrics, predictions) |
| Basic CI/CD | Enhanced workflows with ML model testing |

### Improvements Made:
- ✅ **Real ML Model**: Implemented Random Forest Classifier with scikit-learn
- ✅ **Data Pipeline**: Added data loading, preprocessing, and feature scaling
- ✅ **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score)
- ✅ **Feature Engineering**: StandardScaler for normalization
- ✅ **Model Persistence**: Save/load functionality using joblib
- ✅ **Feature Importance**: Analysis of influential wine characteristics
- ✅ **Robust Testing**: 9 pytest cases + 10 unittest cases

---

## 📁 Project Structure

```
Lab1_New/
│
├── 📂 .github/
│   ├── 📄 pytest_action.yml         # Pytest CI/CD workflow
│   └── 📄 unittest_action.yml       # Unittest CI/CD workflow
│
├── 📂 src/                           # Source code directory
│   ├── 📄 __init__.py               # Package initializer
│   └── 📄 wine_model.py             # Main ML model implementation
│
├── 📂 test/                          # Test directory
│   ├── 📄 __init__.py               # Package initializer
│   ├── 📄 test_pytest.py            # Pytest test suite
│   └── 📄 test_unittest.py          # Unittest test suite
│
├── 📄 .gitignore                     # Git ignore file
├── 📄 README.md                      # Project documentation
└── 📄 requirements.txt               # Project dependencies
```

**Note**: The `data/` directory is not needed as the wine dataset is downloaded directly from the UCI repository URL.

---

## 🔧 Prerequisites

- **Python**: Version 3.8 or higher
- **Git**: For version control
- **GitHub Account**: For repository hosting and CI/CD
- **Operating System**: Windows/Mac/Linux

---

## 💻 Installation Guide

### Step 1: Clone the Repository
```bash
# Clone the repository (replace YOUR_USERNAME with your GitHub username)
git clone https://github.com/YOUR_USERNAME/Lab1_New.git

# Navigate to project directory
cd Lab1_New
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv github_lab1_env

# Activate virtual environment
# On Windows:
github_lab1_env\Scripts\activate

# On Mac/Linux:
source github_lab1_env/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Dependencies Include:
- `pytest==7.4.0` - Testing framework
- `numpy==1.24.3` - Numerical computations
- `pandas==2.0.3` - Data manipulation
- `scikit-learn==1.3.0` - Machine learning algorithms
- `joblib==1.3.1` - Model serialization
- `seaborn==0.12.2` - Statistical visualization
- `matplotlib==3.7.1` - Plotting library

---

## 🏃‍♂️ Running the Project

### 1. Train the Wine Quality Model
```bash
# Run the main model script
python src/wine_model.py
```

**Expected Output:**
```
Training Wine Quality Model...

Model Performance:
accuracy: 0.9313
precision: 0.8621
recall: 0.5814
f1_score: 0.6944

Top 5 Important Features:
alcohol: 0.1734
sulphates: 0.1107
volatile acidity: 0.1049
density: 0.1017
citric acid: 0.0917
```

### 2. Make Predictions (Python Interactive)
```python
# Import the model
from src.wine_model import full_pipeline, load_model

# Train a new model
results = full_pipeline('red')

# Or load existing model
model, scaler = load_model('model.pkl', 'scaler.pkl')

# Make predictions on new data
# Example: [fixed acidity, volatile acidity, citric acid, residual sugar, 
#           chlorides, free sulfur dioxide, total sulfur dioxide, density, 
#           pH, sulphates, alcohol]
new_wine = [[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
prediction = model.predict(scaler.transform(new_wine))
print("Quality:", "Good" if prediction[0] == 1 else "Regular")
```

---

## 🧪 Testing

### Running Pytest Tests
```bash
# Run all pytest tests with verbose output
python -m pytest test/test_pytest.py -v

# Run with coverage report
pytest test/test_pytest.py --cov=src --cov-report=term

# Run specific test
pytest test/test_pytest.py::test_load_wine_data -v
```

**Test Coverage (9 tests):**
- ✅ `test_load_wine_data` - Validates data loading functionality
- ✅ `test_preprocess_data` - Tests data scaling and preprocessing
- ✅ `test_train_model` - Ensures model training works correctly
- ✅ `test_evaluate_model` - Verifies metric calculations
- ✅ `test_predict_wine_quality` - Tests prediction functionality
- ✅ `test_get_feature_importance` - Validates feature importance
- ✅ `test_model_with_different_estimators` - Tests with various parameters

### Running Unittest Tests
```bash
# Run all unittest tests
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestWineModel -v

# Run with discovery
python -m unittest discover test -v
```

**Test Coverage (10 tests):**
- ✅ Data loading and splitting validation
- ✅ Model training and prediction testing
- ✅ Metric evaluation checks
- ✅ Model serialization/deserialization
- ✅ Binary classification validation
- ✅ Feature dimension verification

---

## 🤖 Model Details

### Dataset Information
- **Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type**: Red Wine Quality
- **Samples**: ~1,600 wines
- **Features**: 11 physicochemical properties

### Features Description
| Feature | Description | Range |
|---------|-------------|-------|
| Fixed Acidity | Tartaric acid (g/dm³) | 4.6 - 15.9 |
| Volatile Acidity | Acetic acid (g/dm³) | 0.12 - 1.58 |
| Citric Acid | Freshness and flavor (g/dm³) | 0.0 - 1.0 |
| Residual Sugar | Remaining sugar (g/dm³) | 0.9 - 15.5 |
| Chlorides | Salt content (g/dm³) | 0.012 - 0.611 |
| Free SO₂ | Free sulfur dioxide (mg/dm³) | 1 - 72 |
| Total SO₂ | Total sulfur dioxide (mg/dm³) | 6 - 289 |
| Density | Wine density (g/cm³) | 0.990 - 1.004 |
| pH | Acidity level | 2.74 - 4.01 |
| Sulphates | Wine additive (g/dm³) | 0.33 - 2.0 |
| Alcohol | Alcohol content (% vol) | 8.4 - 14.9 |

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - `n_estimators`: 100 trees
  - `max_depth`: 10
  - `random_state`: 42
- **Preprocessing**: StandardScaler normalization
- **Classification**: Binary (Good vs Regular wine)

### Performance Metrics
| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 93.13% | Overall correct predictions |
| **Precision** | 86.21% | Correct positive predictions |
| **Recall** | 58.14% | True positives identified |
| **F1-Score** | 69.44% | Harmonic mean of precision/recall |

---

## 🔄 GitHub Actions CI/CD

### Automated Workflows

#### 1. Pytest Workflow (`.github/workflows/pytest_action.yml`)
- **Trigger**: Push/PR to main branch
- **Actions**:
  - Sets up Python 3.8 environment
  - Installs dependencies
  - Runs pytest with XML reporting
  - Uploads test artifacts
  - Reports success/failure

#### 2. Unittest Workflow (`.github/workflows/unittest_action.yml`)
- **Trigger**: Push/PR to main branch
- **Actions**:
  - Sets up Python 3.8 environment
  - Installs dependencies
  - Runs unittest suite
  - Generates test reports
  - Notifies test results

### Viewing CI/CD Results
1. Go to your GitHub repository
2. Click on the **"Actions"** tab
3. View workflow runs and their status
4. Click on any run for detailed logs

---

## 📊 Results

### Model Performance Summary
```
Training Samples: 1,279
Testing Samples: 320
Training Time: ~2 seconds
Prediction Time: <1ms per sample
Model Size: ~500KB
```

### Feature Importance (Top 5)
1. **Alcohol** (17.34%) - Most influential factor
2. **Sulphates** (11.07%) - Wine preservative
3. **Volatile Acidity** (10.49%) - Vinegar taste
4. **Density** (10.17%) - Wine body
5. **Citric Acid** (9.17%) - Freshness factor

---

## 🚦 Quick Start Guide

```bash
# 1. Clone and enter directory
git clone https://github.com/YOUR_USERNAME/Lab1_New.git && cd Lab1_New

# 2. Setup environment
python -m venv env && source env/bin/activate  # Mac/Linux
python -m venv env && env\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the model
python src/wine_model.py

# 5. Run tests
pytest test/ -v
```

---

## 📝 License

This project is part of an academic assignment for MLOps course.

---