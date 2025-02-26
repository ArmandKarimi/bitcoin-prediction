#!/bin/bash

# Update and install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize Airflow (first-time setup)
export AIRFLOW_HOME=~/airflow
airflow db init

# Start Airflow webserver and scheduler
airflow scheduler -D
airflow webserver -D

echo "✅ Airflow & DBT setup completed!"
