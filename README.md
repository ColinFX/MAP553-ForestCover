# MAP553-ForestCover

MAP553 kaggle project to predict forest cover type from cartographic variables.

Performance of MXGBClassifier on Kaggle leaderboard

```text
params = {
    "weight": 0.5,
    "n_estimators": 125,
    "max_depth": 10,
    "learning_rate": 0.1,
    "gamma": 0,
} -> 0.74993
    
params = {
    "weight": 0.8,
    "n_estimators": 50,
    "max_depth": 10,
    "learning_rate": 0.1,
    "gamma": 1,
} -> 0.72694

params = {
    "weight": 1,
    "n_estimators": 50,
    "max_depth": 10,
    "learning_rate": 0.1,
    "gamma": 0.1,
} -> 0.72826

params = {
    "weight": 1,
    "n_estimators": 30,
    "max_depth": 4,
    "learning_rate": 1,
    "gamma": 0,
} -> 0.69184
```
