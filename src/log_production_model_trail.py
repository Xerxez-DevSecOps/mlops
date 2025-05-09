from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os


def log_production_model(config_path):
    # Read configuration
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)

    # Fetch runs and find the lowest MAE
    runs = mlflow.search_runs(experiment_ids=None)  # Search across all experiments
    print("Runs DataFrame:")
    print(runs.head())  # Debugging: Print the structure of the runs DataFrame

    if "metrics.mae" not in runs.columns:
        print("Available columns in runs DataFrame:", runs.columns)  # Debugging: Show available columns
        raise KeyError("The 'metrics.mae' column is missing in the runs DataFrame.")

    lowest_mae = runs["metrics.mae"].min()
    lowest_run_id = runs[runs["metrics.mae"] == lowest_mae]["run_id"].iloc[0]

    # Transition model versions
    client = MlflowClient()
    logged_model = None
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")

    # Load and save the model
    if logged_model:
        load_model = mlflow.pyfunc.load_model(logged_model)
        model_path = config["model_dirs"]
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, "model.pkl")
        joblib.dump(load_model, model_file)
        print(f"Model saved to {model_file}")
    else:
        raise ValueError("No logged model found for the lowest run ID.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)