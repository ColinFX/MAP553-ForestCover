"""
Implementation of Multi-XGBoost Classifier.
"""

import csv
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

from typing import List, Dict


warnings.filterwarnings("ignore", category=UserWarning)


def soil_type_2_elu(soil_type: int) -> int:
    assert 0 < soil_type < 41, \
        "Soil type out of boundary 1~40."
    code_dict = [
        2702, 2703, 2704, 2705, 2706,
        2717, 3501, 3502, 4201, 4703,
        4704, 4744, 4758, 5101, 5151,
        6101, 6102, 6731, 7101, 7102,
        7103, 7201, 7202, 7700, 7701,
        7702, 7709, 7710, 7745, 7746,
        7755, 7756, 7757, 7790, 8703,
        8707, 8708, 8771, 8772, 8776,
        ]
    return code_dict[soil_type - 1]


def get_climatic_zone(elu: int) -> int:
    res = elu // 1000
    assert 0 < res <= 8, "Climatic zone code out of boundary 1~8."
    return res


def get_geologic_zone(elu: int) -> int:
    res = elu % 1000 // 100
    assert 0 < res <= 8, "Geologic zone code out of boundary 1~8."
    return res


def get_third_digit(elu: int) -> int:
    return elu % 100 // 10


def get_fourth_digit(elu: int) -> int:
    return elu % 10


def preprocess(df: pd.DataFrame, mode: str = "train") -> List[np.ndarray]:
    """
    Preprocess the dataframe and return [X, y] without reshuffling nor rescaling.
    X is of shape (n,d+2) and y is of shape (n).
    The first column of X contains the ID of each record,
    whilst the second column contains the area code of each record.
    :param mode whether "train" or "test"
    """

    # soil types
    df.insert(loc=0, column="Soil_Type", value=0)
    for i in range(1, 41):
        column_name = "Soil_Type" + str(i)
        df.loc[df[column_name] == 1, "Soil_Type"] = i
        df.drop(column_name, axis=1, inplace=True)

    df["elu"] = [soil_type_2_elu(i) for i in df["Soil_Type"]]
    df.drop("Soil_Type", axis=1, inplace=True)

    df["climatic_zone"] = [get_climatic_zone(i) for i in df["elu"]]
    df["geologic_zone"] = [get_geologic_zone(i) for i in df["elu"]]
    df["third_digit"] = [get_third_digit(i) for i in df["elu"]]
    df["fourth_digit"] = [get_fourth_digit(i) for i in df["elu"]]
    df.drop("elu", axis=1, inplace=True)

    # wilderness area
    df.insert(loc=0, column="Wilderness_Area", value=0)
    for i in range(1, 5):
        column_name = "Wilderness_Area" + str(i)
        df.loc[df[column_name] == 1, "Wilderness_Area"] = i
        df.drop(column_name, axis=1, inplace=True)

    # reformat
    ids = df["Id"].to_numpy()
    df.drop("Id", axis=1, inplace=True)

    if mode == "train":
        y = df["Cover_Type"].to_numpy()
        df.drop("Cover_Type", axis=1, inplace=True)
    elif mode == "test":
        y = None
    else:
        raise AssertionError("Unexpected mode, try \"train\" or \"test\" instead. ")

    areas = df["Wilderness_Area"].to_numpy()
    df.drop("Wilderness_Area", axis=1, inplace=True)

    X = df.to_numpy()

    # one hot encode
    enc = OneHotEncoder(categories=[np.arange(1, 9), np.arange(1, 9), np.arange(0, 10), np.arange(0, 10)])
    enc.fit(X[:, -4:])

    X = np.concatenate((
        ids.reshape(-1, 1),
        areas.reshape(-1, 1),
        X[:, :-4],
        enc.transform(X[:, -4:]).toarray()
    ), axis=1)

    return [X, y]


class MXGBClassifier(object):
    def __init__(self, params: Dict[str, float]):
        """
        The classifier is expecting X of shape (n,d+2), containing columns of ID and of area.
        param weight: the weight of the general model, expecting a float between 0 and 1.
        """
        self.weight = params["weight"]
        assert 0 <= self.weight <= 1, "weight out of boundary [0,1]."
        self.general_model = XGBClassifier(n_jobs=-1, verbosity=0)
        self.general_model.set_params(**params)
        self.area_models = [XGBClassifier(n_jobs=-1, verbosity=0) for _ in range(4)]
        for area_model in self.area_models:
            area_model.set_params(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.general_model.fit(X[:, 2:], y)
        for i in range(4):
            mask = (X[:, 1] == i + 1)
            Xi = X[mask, 2:]
            yi = y[mask]
            self.area_models[i].fit(Xi, yi)

    def predict(self, X: np.ndarray) -> np.ndarray:
        general_proba = self.general_model.predict_proba(X[:, 2:])
        area_proba = np.zeros(general_proba.shape)
        for i in range(4):
            classes = np.array([j + 1 in self.area_models[i].classes_ for j in range(7)])
            mask = (X[:, 1] == i + 1)
            area_proba[mask][:, classes] += self.area_models[i].predict_proba(X[mask, 2:])
        final_proba = self.weight * general_proba + (1 - self.weight) * area_proba
        return (np.argmax(final_proba, axis=1) + 1).astype(int)


if __name__ == "__main__":
    df_train = pd.read_csv("./data/train.csv")
    X, y = preprocess(df_train)

    # TUNING

    # params_list = {
    #     "weight": [0, 1],
    #     "n_estimators": [50],
    #     "max_depth": [10],
    #     "learning_rate": [0.1],
    #     "gamma": [0.1],
    # }
    #
    # with open("mxgb_result.csv", "a") as file:
    #     writer = csv.DictWriter(file, fieldnames=list(params_list.keys())+["train_acc", "val_acc"])
    #     # writer.writeheader()
    #
    #     for params in tqdm(ParameterGrid(params_list)):
    #         kf = KFold(n_splits=5, shuffle=True)
    #         train_accuracy = []
    #         val_accuracy = []
    #
    #         for i, (train_index, val_index) in enumerate(kf.split(X)):
    #             X_train = X[train_index]
    #             X_val = X[val_index]
    #             y_train = y[train_index]
    #             y_val = y[val_index]
    #
    #             classifier = MXGBClassifier(params)
    #             classifier.fit(X_train, y_train)
    #             y_train_pred = classifier.predict(X_train)
    #             train_accuracy.append(accuracy_score(y_train, y_train_pred))
    #             y_val_pred = classifier.predict(X_val)
    #             val_accuracy.append(accuracy_score(y_val, y_val_pred))
    #
    #         result = params.copy()
    #         result["train_acc"] = np.mean(train_accuracy)
    #         result["val_acc"] = np.mean(val_accuracy)
    #
    #         writer.writerow(result)

    # PREDICTING

    params = {
        "weight": 1,
        "n_estimators": 30,
        "max_depth": 4,
        "learning_rate": 1,
        "gamma": 0,
    }

    classifier = MXGBClassifier(params)
    classifier.fit(X, y)

    df_test = pd.read_csv("./data/test-full.csv")
    X_test, _ = preprocess(df_test, mode="test")
    ids = [int(id) for id in X_test[:, 0]]

    y_test = classifier.predict(X_test)
    df_result = pd.DataFrame(list(zip(ids, y_test)), columns=['Id', 'Cover_Type'])
    df_result.to_csv("./data/mxgb_pred.csv", index=False)
