from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'Airflow_Lab1_WineQuality',
    default_args=default_args,
    description='Wine Quality Classification with Random Forest',
    schedule_interval=None,  # Manual trigger
    catchup=False,
)

# Task 1: Load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess data
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Build and save Random Forest model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "wine_model.pkl"],
    provide_context=True,
    dag=dag,
)

# Task 4: Load model and show analysis
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["wine_model.pkl", build_save_model_task.output],
    dag=dag,
)

# Set task dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.cli()
