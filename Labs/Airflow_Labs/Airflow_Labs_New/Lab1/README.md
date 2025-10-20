# Lab 1 Execution Guide - Screenshots and Results

## 1. Airflow Login Page

<img width="1920" height="1080" alt="Screenshot (267)" src="https://github.com/user-attachments/assets/1773e75c-cc95-429e-9842-8b62dfd4047c" />


**Description:** Access Airflow at `localhost:8080` and login with credentials:
- Username: `airflow2`
- Password: `airflow2`

This page appears when you first navigate to the Airflow web interface. After entering credentials, you're redirected to the main DAGs dashboard.

---

## 2. DAGs List View

![DAGs List](screenshots/2_dag_list.png)

**Description:** The Airflow dashboard displaying all available DAGs:
- **Airflow_Lab1** - Main K-Means clustering DAG (owner: your_name)
- **test_dag** - Test DAG created during troubleshooting

Shows:
- Total DAGs: 2
- Status indicators (All, Active, Paused, Running, Failed)
- Last run information
- Quick action buttons (trigger, delete)

---

## 3. DAG Summary Page

![DAG Summary](screenshots/3_dag_details.png)

**Description:** Clicking on Airflow_Lab1 shows the DAG summary with:
- Total Tasks: 4
- All tasks are PythonOperators
- Schedule: None (manual triggering only)
- Description: "Dag example for Lab 1 of Airflow series"
- Owner: your_name
- DAG ID: Airflow_Lab1

This page provides an overview before running the DAG.

---

## 4. DAG Execution Graph - All Tasks Successful

![DAG Execution](screenshots/4_dag_execution_graph.png)

**Description:** Graph view showing complete workflow execution in sequence:

1. **load_data_task** (Green ✓)
   - Loads data from CSV file
   - Serializes using pickle
   - Output: Serialized DataFrame

2. **data_preprocessing_task** (Green ✓)
   - Deserializes input data
   - Removes missing values
   - Selects features: BALANCE, PURCHASES, CREDIT_LIMIT
   - Applies MinMaxScaler normalization
   - Output: Normalized features

3. **build_save_model_task** (Green ✓)
   - Trains K-Means models for clusters 1-50
   - Calculates SSE (Sum of Squared Errors)
   - Saves model to `dags/model/model.sav`
   - Output: SSE values for elbow analysis

4. **load_model_task** (Green ✓)
   - Loads saved K-Means model
   - Applies KneeLocator for elbow method
   - Makes predictions on test data
   - Logs optimal cluster count

**Execution Summary:**
- Total Duration: 00:00:20 (20 seconds)
- Status: All tasks completed successfully
- Color: Green boxes indicate success
- All dependencies met and executed in proper sequence

---

## 5. Results - Optimal Clusters Found

![Optimal Clusters Result](screenshots/5_optimal_clusters_result.png)

**Description:** Task logs from the final `load_model_task` showing the critical result:

```
[2025-10-20, 17:31:36 UTC] (logging_mixin.py:188) INFO - Optimal no. of clusters: 8
```

**Key Information Extracted from Logs:**
- Task: load_model_task
- Attempt: 1
- Status: SUCCESS
- Optimal number of clusters determined: **8**
- Test prediction value: 5
- Method: KneeLocator with elbow method analysis

This is the main deliverable of Lab 1 - determining that 8 is the optimal number of clusters for the advertising dataset using K-Means clustering and the elbow method.

---

## Workflow Summary Table

| Step | Task Name | Input | Processing | Output | Status |
|------|-----------|-------|-----------|--------|--------|
| 1 | load_data_task | file.csv | Read CSV, serialize | Serialized DataFrame | ✓ Success |
| 2 | data_preprocessing_task | Serialized data | Normalize features | Normalized array | ✓ Success |
| 3 | build_save_model_task | Normalized data | Train K-Means 1-50 | SSE values + model | ✓ Success |
| 4 | load_model_task | SSE + model | Elbow method | **Clusters: 8** | ✓ Success |

---

## Execution Timeline

**October 20, 2025, 17:31:16 UTC**

- **17:31:16** - DAG triggered manually
- **17:31:35** - load_data_task completed (5 seconds)
- **17:31:36** - data_preprocessing_task completed (1 second)
- **17:31:36** - build_save_model_task completed (0 seconds)
- **17:31:36** - load_model_task completed (0 seconds)
- **Total Duration:** 20 seconds
- **Final Status:** SUCCESS

---

## Key Findings and Results

**Primary Result:** Optimal number of clusters = **8**

**Dataset Information:**
- Source: Advertising dataset
- Features used: BALANCE, PURCHASES, CREDIT_LIMIT
- Training samples: Loaded from file.csv
- Test samples: Loaded from test.csv

**K-Means Configuration:**
- Cluster range tested: 1-50
- Initialization method: Random (10 init)
- Maximum iterations: 300
- Random state: 42 (reproducible)

**Elbow Method:**
- Tool: KneeLocator library
- Curve type: Convex
- Direction: Decreasing
- Result: Knee point identified at cluster count = 8

**Model Artifacts:**
- Saved model: `dags/model/model.sav`
- Model type: Trained K-Means object with 8 clusters
- Test prediction: First test sample assigned to cluster 5

---

## How to Reproduce Lab 1

### Prerequisites
- Docker Desktop installed and running
- 4GB+ RAM allocated to Docker
- Working directory: C:\airflow_work\Lab_1

### Steps

1. **Start Airflow**
   ```
   docker compose up
   ```
   Wait 2-3 minutes for services to start.

2. **Access Web Interface**
   - Open browser to http://localhost:8080
   - Login with airflow2/airflow2

3. **Trigger the DAG**
   - Navigate to DAGs page
   - Find "Airflow_Lab1"
   - Click the play button (▶)
   - Click "Trigger" to confirm

4. **Monitor Execution**
   - Click on Airflow_Lab1
   - Go to "Graph" tab
   - Watch tasks turn green as they complete

5. **View Results**
   - Click on load_model_task (rightmost box)
   - Click "Logs" tab
   - Look for "Optimal no. of clusters: 8"

6. **Stop Airflow**
   ```
   docker compose down
   ```

---

## Troubleshooting During Execution

**Issue:** Tasks showing as running but not completing
- Solution: Wait 20+ seconds, as K-Means training for 50 clusters takes time

**Issue:** Logs not appearing
- Solution: Refresh browser (Ctrl+Shift+R) and click on task again

**Issue:** Model file not created
- Solution: Check that build_save_model_task completed successfully (should be green)

**Issue:** Cluster count not appearing in logs
- Solution: Ensure load_model_task is green, then scroll down in logs panel

---

## Lab 1 Completion Checklist

- [x] Docker and Airflow installed and configured
- [x] DAG file created with 4 tasks
- [x] All tasks executed successfully
- [x] No errors in task logs
- [x] Model file saved to dags/model/model.sav
- [x] Optimal clusters determined: 8
- [x] Results logged and visible in Airflow UI
- [x] Documentation completed
- [x] Code pushed to GitHub

---

## Files Generated During Execution

After running the DAG, the following files are created:

- `dags/model/model.sav` - Trained K-Means model with 8 clusters
- `logs/dag_id=Airflow_Lab1/...` - Execution logs for each task
- `working_data/` - Any intermediate data (if configured)

---

## Next Steps

- Proceed to **Lab 2** for email notifications and Flask API monitoring
- Or make modifications to Lab 1 as part of your contribution

---

**Lab Status:** COMPLETED SUCCESSFULLY ✓

**Date Completed:** October 20, 2025

**Result Achieved:** Optimal K-Means clusters = 8

**All Tasks:** Passed

---
