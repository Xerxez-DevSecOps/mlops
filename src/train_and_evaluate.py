import os
import yaml
import pandas as pd
import argparse 
from pkgutil import get_data
from get_data import get_data, read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet
import joblib
import json
import numpy as np
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    model_dir = config["model_dirs"]

    alpha = config["estimator"]["ElasticNet"]["alpha"]
    l1_ratio = config["estimator"]["ElasticNet"]["l1_ratio"]

    target = config["base"]["target_col"]
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    train_y = train[target]
    test_y = test[target]

    #################CREATE MODEL#######################

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("R2: ", r2)

    score_files = config["reports"]["scores"]
    params_files = config["reports"]["params"]

    with open(score_files, "w") as f:
        scores = {
            "rmse" : rmse,
            "mae" : mae,
            "r2" : r2
        }
        json.dump(scores, f, indent=4)

    with open(params_files, "w") as f:
        scores = {
            "alpha" : alpha,
            "l1_ratio" : l1_ratio,
        }
        json.dump(params_files, f, indent=4)


        ##############SAVE MODEL###########################
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(lr, model_path)
    print("Model saved to ", model_path)

  ######################PREDICTION #######################

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)