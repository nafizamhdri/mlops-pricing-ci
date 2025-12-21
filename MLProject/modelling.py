import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import estimator_html_repr
import json
import os


def main():
    print("Starting Skilled Model Training (Hyperparameter Tuning)")
    
    # 0. Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Set MLflow URI and Experiment (using absolute path)
    mlflow_db_path = os.path.join(script_dir, 'mlflow.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("Used Car Price Prediction")
    
    # 2. Load Data
    data_path = os.path.join(script_dir, 'car_data_processed2.csv')

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}!")
        return

    df = pd.read_csv(data_path)
    
    # 2. Split Data
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    # 4. Start MLflow Run (or use existing if called by MLflow Projects)
    # Check if there's already an active run (e.g., from MLflow Projects)
    active_run = mlflow.active_run()
    
    if active_run is None:
        # No active run - create one (standalone execution)
        run = mlflow.start_run(run_name="Skilled_Hyperparameter_Tuning")
        should_end_run = True
    else:
        # Active run exists (MLflow Projects execution) - use it
        run = active_run
        should_end_run = False
        print(f"Using existing MLflow run: {active_run.info.run_id}")
    
    try:
        print("Tuning Hyperparameters with GridSearchCV")
        
        # Grid Search
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Params: {best_params}")
        
        # --- LOGGING ---
        
        # A. Log Parameters available in the best model
        mlflow.log_params(best_params)
        
        # B. Predict and Evaluate
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        
        # C. Log Metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        
        # D. Log Model
        print("Logging model to MLflow...")
        try:
            # Use simpler syntax - mlflow.sklearn.log_model(model, path)
            mlflow.sklearn.log_model(best_model, "model")
            print(" Model logged successfully")
        except Exception as e:
            print(f" Error logging model: {e}")
            import traceback
            traceback.print_exc()
        
        # E. Create and Log Artifacts
        
        # 1. Estimator HTML
        print("Generating estimator.html")
        estimator_path = os.path.join(script_dir, "estimator.html")
        with open(estimator_path, "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(best_model))
        mlflow.log_artifact(estimator_path)

        # 2. Metric Info JSON
        print("Generating metric_info.json")
        metric_path = os.path.join(script_dir, "metric_info.json")
        metrics_dict = {"MAE": mae, "MSE": mse, "R2": r2}
        with open(metric_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact(metric_path)

        # 3. Residual Plot 
        print("Generating training_confusion_matrix.png (Residual Plot for Regression)")
        plot_path = os.path.join(script_dir, "training_confusion_matrix.png")
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        print(f"Run Complete. Check http://127.0.0.1:5000/ . Artifacts saved to {script_dir}")
    
    finally:
        # End run only if we created it (standalone execution)
        if should_end_run:
            mlflow.end_run()

if __name__ == "__main__":
    main()
