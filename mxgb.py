import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from typing import List


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


def preprocess(df: pd.DataFrame) -> List[np.ndarray]:
    """
    Preprocess the dataframe and return [X, y] without reshuffling nor rescaling.
    X is of shape (n,d+2) and y is of shape (n).
    The first column of X contains the Id of each record,
    whilst the second column contains the area code of each record.
    """

    # soil types
    df.insert(loc=0, column="Soil_Type", value=0)
    for i in range(1, 41):
        column_name = "Soil_Type" + str(i)
        df.loc[df[column_name] == 1, "Soil_Type"] = i
        df.drop(column_name, axis=1, inplace=True)

    df.insert(loc=0, column="elu", value=[soil_type_2_elu(i) for i in df["Soil_Type"]])
    df.drop("Soil_Type", axis=1, inplace=True)

    df.insert(loc=0, column="climatic_zone", value=[get_climatic_zone(i) for i in df["elu"]])
    df.insert(loc=1, column="geologic_zone", value=[get_geologic_zone(i) for i in df["elu"]])
    df.insert(loc=2, column="third_digit", value=[get_third_digit(i) for i in df["elu"]])
    df.insert(loc=3, column="fourth_digit", value=[get_fourth_digit(i) for i in df["elu"]])
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

    y = df["Cover_Type"].to_numpy()
    df.drop("Cover_Type", axis=1, inplace=True)

    areas = df["Wilderness_Area"].to_numpy()
    df.drop("Wilderness_Area", axis=1, inplace=True)

    X = df.to_numpy()

    # one hot encode
    enc = OneHotEncoder(categories=[np.arange(1, 9), np.arange(1, 9), np.arange(0, 10), np.arange(0, 10)])
    enc.fit(X[:, :4])

    X = np.concatenate((ids.reshape(-1, 1), areas.reshape(-1, 1), enc.transform(X[:, :4]).toarray(), X[:, 4:]), axis=1)

    return [X, y]


class MXGBClassifier(object):
    def __init__(self, weight: float = 0.5):
        """
        param weight: the weight of the general model, expecting a float between 0 and 1.
        """
        assert 0 <= weight <= 1, "weight out of boundary [0,1]."
        self.weight = weight
        self.general_model = XGBClassifier(eval_metric="mlogloss")
        self.area_models = [XGBClassifier(eval_metric="mlogloss") for i in range(4)]

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
            area_proba[mask][:, classes] = self.area_models[i].predict_proba(X[mask, 2:])
        final_proba = self.weight * general_proba + (1 - self.weight) * area_proba
        return (np.argmax(final_proba, axis=1) + 1).astype(int)


if __name__ == "__main__":
    df_train = pd.read_csv("./data/train.csv")
    X, y = preprocess(df_train)
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    classifier = MXGBClassifier()
    classifier.fit(X_train, y_train)

    y_train_pred = classifier.predict(X_train)
    print("# TRAIN # \n\n", confusion_matrix(y_train, y_train_pred))
    print("\n", classification_report(y_train, y_train_pred))

    y_val_pred = classifier.predict(X_val)
    print("# VALIDATION # \n\n", confusion_matrix(y_val, y_val_pred))
    print("\n", classification_report(y_val, y_val_pred))
