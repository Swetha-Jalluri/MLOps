# Airflow Lab 1 - Enhanced Version

## Overview
This is an enhanced version of the original Airflow Lab 1. The lab has been modified to implement a **supervised machine learning workflow** using Random Forest classification on the Wine Quality dataset, replacing the original unsupervised K-Means clustering approach.


<img width="1901" height="931" alt="Screenshot (297)" src="https://github.com/user-attachments/assets/1d768479-4051-4804-afc0-d63545286010" />


## Modifications Made

### 1. Dataset Change
**Original:** Generic tabular data for clustering  
**Modified:** UCI Wine Quality Dataset
- **Features:** 11 features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **Samples:** 1,599 wine samples
- **Target:** Quality score (0-10)
- **Task:** Binary classification (Good wine vs Bad wine, threshold: quality >= 6)

### 2. Machine Learning Model Change
**Original:** K-Means Clustering (Unsupervised)  
**Modified:** Random Forest Classifier (Supervised)
- **Algorithm:** Random Forest with 100 decision trees
- **Max Depth:** 10 levels
- **Problem Type:** Supervised learning (binary classification)
- **Train-Test Split:** 80-20

### 3. Output Metrics
**Original:**
- SSE (Sum of Squared Errors) values
- Optimal number of clusters using Elbow Method

**Modified:**
- Training Accuracy
- Test Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Ranking

## Workflow Architecture

```
┌─────────────────────┐
│   load_data_task    │
│ (Load Wine Quality) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│  data_preprocessing_task        │
│  (Scale & Binary Classification)│
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│   build_save_model_task          │
│  (Train Random Forest & Evaluate)│
└──────────┬───────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│    load_model_task             │
│  (Feature Importance Analysis) │
└────────────────────────────────┘
```

## Files Modified

### 1. `dags/src/lab.py`
Complete rewrite with 4 functions:
- `load_data()` - Loads Wine Quality dataset
- `data_preprocessing()` - Scales features and converts to binary classification (Good/Bad wine)
- `build_save_model()` - Trains Random Forest and evaluates performance
- `load_model_elbow()` - Analyzes feature importance for wine quality prediction

### 2. `dags/airflow.py`
Updated DAG definition:
- DAG name: `Airflow_Lab1_WineQuality`
- 4 sequential tasks with dependencies
- XCom enabled for data passing between tasks
- Manual trigger (no scheduling)

### 3. `docker-compose.yaml`
Updated configuration:
- `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'`
- Added Python packages: `pandas scikit-learn kneed joblib`
- `AIRFLOW__CORE__ENABLE_XCOM_PICKLING: 'true'`
- Updated web credentials: `airflow2/airflow2`

## How to Run

### Prerequisites
- Docker Desktop installed and running
- Minimum 4GB RAM allocated to Docker (8GB recommended)

### Setup and Execution

1. **Navigate to the lab directory:**
```bash
cd Lab1
```

2. **Start Airflow with Docker:**
```bash
docker compose up
```

3. **Wait for initialization** (1-2 minutes until you see):
```
airflow-scheduler-1 | 127.0.0.1 - - [...] "GET /health HTTP/1.1" 200 -
```

4. **Access Airflow UI:**
```
http://localhost:8081
```

5. **Login credentials:**
- Username: `airflow2`
- Password: `airflow2`

6. **Trigger the DAG:**
- Find `Airflow_Lab1_WineQuality` in the DAG list
- Toggle the DAG to "On" (enable it)
- Click "Trigger DAG" button
- Monitor execution in the Graph view

7. **View Results:**
- Go to Graph tab
- Click on `load_model_task` (last task)
- Click "Logs" tab
- Scroll to see model accuracy and feature importance rankings

### Stopping Airflow
```bash
docker compose down
```

## Project Structure

```
Lab1/
├── dags/
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py                 # Modified - Wine Quality + Random Forest
│   ├── airflow.py                 # Modified - New DAG
│   ├── data/                       # Wine quality data (if local)
│   └── model/                      # Generated model files
├── logs/                           # Execution logs
├── config/                         # Configuration
├── plugins/                        # Airflow plugins
├── docker-compose.yaml             # Modified - Updated config
├── .env                            # Environment variables
└── README.md                       # This file
```

## Key Improvements Over Original

| Aspect | Original | Enhanced |
|--------|----------|----------|
| **Problem Type** | Unsupervised (Clustering) | Supervised (Classification) |
| **Dataset** | Generic data | Wine Quality (1,599 samples) |
| **Model** | K-Means | Random Forest |
| **Output Metrics** | Cluster count | Accuracy, Precision, Recall, F1 |
| **Feature Analysis** | Cluster centers | Feature importance for wine quality |
| **Real-world Use** | Grouping similar items | Predicting wine quality |
| **Interpretability** | Low | High |

## Model Performance

The Random Forest classifier trained on Wine Quality dataset provides:
- Binary classification of wine quality (Good vs Bad)
- Identification of most influential factors in wine quality
- Clear feature importance insights (alcohol, acidity levels, etc.)
- Robust evaluation metrics (accuracy, precision, recall)

## Technologies Used
- **Apache Airflow 2.9.2** - Workflow orchestration
- **Python 3.12** - Programming language
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **Docker** - Containerization

## Conclusion
This enhanced version demonstrates a practical transition from exploratory data analysis (clustering) to predictive analytics (classification) using the Wine Quality dataset. The workflow showcases best practices in MLOps including proper train-test splitting, model evaluation, and feature importance analysis.

