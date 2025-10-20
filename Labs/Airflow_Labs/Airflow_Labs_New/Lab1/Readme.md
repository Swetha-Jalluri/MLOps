# Airflow Lab 1: K-Means Clustering with Elbow Method

## Overview

This project implements an Apache Airflow DAG pipeline that automates a machine learning workflow for K-Means clustering analysis. The pipeline loads data, preprocesses it, builds a clustering model, and determines the optimal number of clusters using the elbow method.

**Result: Optimal number of clusters determined = 8**

## Key Features

- Docker-based Airflow Setup: Runs on Windows, macOS, and Linux without additional installation
- Automated ML Workflow: Four-task DAG with proper dependency management
- K-Means Clustering: Builds a model with configurable cluster range (1-50)
- Elbow Method: Automatically determines optimal clusters using KneeLocator
- Airflow Web UI: Monitor and manage DAG execution through intuitive interface

## Project Structure
```
C:\airflow_work\Lab_1\
├── dags/
│   ├── data/
│   │   ├── file.csv
│   │   └── test.csv
│   ├── model/
│   │   └── model.sav (generated after run)
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py
│   └── airflow.py
├── logs/
├── plugins/
├── config/
├── docker-compose.yaml
├── .env
└── README.md
```

## Prerequisites

- Docker Desktop (version 28.5.1 or higher)
- 4GB minimum memory allocated to Docker (8GB recommended)
- Windows, macOS, or Linux

## Setup Instructions

### 1. Create Working Directory
```powershell
cd C:\
mkdir airflow_work
cd airflow_work
```

### 2. Download Docker Compose
```powershell
curl -o docker-compose.yaml https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml
```

### 3. Create Required Directories
```powershell
mkdir dags, logs, plugins, config
```

### 4. Create .env File
```powershell
"AIRFLOW_UID=50000" | Out-File -FilePath .env -Encoding UTF8
```

### 5. Configure docker-compose.yaml

Update these settings:

**Disable Examples:**
```yaml
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
```

**Add Python Packages:**
```yaml
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn kneed}
```

**Add Volume:**
```yaml
- ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data
```

**Change Admin Credentials:**
```yaml
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
```

### 6. Initialize Database
```powershell
docker compose up airflow-init
```

### 7. Start Airflow
```powershell
docker compose up
```

## Accessing Airflow

1. Open browser to: http://localhost:8080
2. Login with:
   - Username: airflow2
   - Password: airflow2

## DAG Workflow

### Task 1: load_data_task
- Loads data from CSV file
- Serializes using pickle

### Task 2: data_preprocessing_task
- Removes missing values
- Selects features: BALANCE, PURCHASES, CREDIT_LIMIT
- Applies MinMaxScaler normalization

### Task 3: build_save_model_task
- Trains K-Means models (1-50 clusters)
- Calculates SSE values
- Saves model to file

### Task 4: load_model_task
- Loads saved model
- Uses KneeLocator for elbow method
- Determines optimal clusters: **8**

## Running the DAG

1. Navigate to DAGs page
2. Find Airflow_Lab1
3. Click play button to trigger
4. Monitor execution in Graph view

## Viewing Results

1. Click load_model_task
2. Click Logs tab
3. Find: "Optimal no. of clusters: 8"

## ML Functions

### load_data()
Loads CSV and returns serialized DataFrame

### data_preprocessing(data)
Preprocesses and normalizes data

### build_save_model(data, filename)
Trains K-Means and saves model, returns SSE values

### load_model_elbow(filename, sse)
Determines optimal clusters using elbow method

## Dependencies

- pandas
- scikit-learn
- kneed
- pickle
- Apache Airflow 2.9.2

## Stopping Airflow
```powershell
docker compose down
```

## Success Indicators

✓ All 4 tasks execute successfully (green boxes)
✓ Execution time: ~20 seconds
✓ No errors in logs
✓ "Optimal no. of clusters: 8" logged
✓ Model file created at dags/model/model.sav

## References

- Apache Airflow Documentation: https://airflow.apache.org/docs/
- Scikit-learn K-Means: https://scikit-learn.org/
- Docker Airflow: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/

---

Lab Status: Complete
Last Run: October 20, 2025
Optimal Clusters Found: 8
