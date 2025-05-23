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

def eval_metrics(actul, pred):
    rmse = np.sqrt(mean_squared_error(actul,pred))
    mae = mean_absolute_error(actul, pred)
    r2 = r2_score(actul, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
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

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    #print("ElasticNet Model (alpha = %f, l1_ratio=%f):" %(alpha, l1_ratio))

    #print("RMSE:%s" %rmse)
    #print("MAE:%s" %mae)
    #print("R2 Score:%s" %r2)

    score_files = config["reports"]["score"]
    params_files = config["reports"]["params"]

    with open(score_files,"w") as f:
        scores = {
            "rmse": rmse,
            "mae" : mae,
            "r2" : r2
        }
        json.dump(scores, f, indent=4)

    with open(params_files,"w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio" : l1_ratio,
        }
        json.dump(params, f, indent=4)

    ################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)