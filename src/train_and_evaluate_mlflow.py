import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import get_data, read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import json
import numpy as np
import mlflow
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import argparse
import logging
from mlflow.server import get_app_client
#from mlflow.server.auth.client import AuthServiceClient

def eval_metrics(actul, pred):
    rmse = np.sqrt(mean_squared_error(actul,pred))
    mae = mean_absolute_error(actul, pred)
    r2 = r2_score(actul, pred)
    return rmse, mae, r2

##################AUTHENTICATION#####################

#import requests

#response = requests.post(
#    "http://127.0.0.1:5000",
#    json={
#        "username": "admin",
#        "password": "admin",
#    },
#)

###############################
def train_and_evaluate_mlops(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dirs"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ration"]

    target = config["base"]["target_col"]
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    train_y = train[target]
    test_y = test[target]

    ########################################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    #########################################
    #auth_client = get_app_client("basic-auth", tracking_uri=remote_server_uri)
    #auth_client = mlflow.server.get_app_client("basic-auth", tracking_uri="http://127.0.0.1:5000")
    #auth_client.create_user(username="admin", password="admin")
    #auth_client.update_user_admin(username="admin", is_admin=True)
    ####################################

    mlflow.set_experiment(mlflow_config["experiment_name"])

    

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ration", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store !="file" :
            mlflow.sklearn.log_model(lr, "model", registered_model_name = mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(lr, "model")  #, signature=signature 
 
    ################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate_mlops(config_path=parsed_args.config)