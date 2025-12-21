import pandas as pd
import numpy as np
import mlflow
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("Starting Model Training with MLflow Autolog")
    
    # 0. Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Set MLflow URI and Experiment (using absolute path)
    mlflow_db_path = os.path.join(script_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("Used Car Price Prediction - Autolog")
    
    # 2. Enable MLflow Autolog (Standard)
    mlflow.autolog()
    
    # 3. Load Data
    data_path = os.path.join(script_dir, 'car_data_processed2.csv')
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}!")
        return

    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # 2. Split Data
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # 3. Train Model with MLflow Autolog
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        print("Training RandomForestRegressor")
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model - autolog will capture everything
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics (will be auto-logged)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        

if __name__ == "__main__":
    main()
