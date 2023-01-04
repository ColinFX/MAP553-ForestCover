# MAP553-ForestCover

MAP553 kaggle project to predict forest cover type from cartographic variables.

Current performance on the split validation set: 

```
# VALIDATION # 

 [[360  73   0   0  12   0  41]
 [107 359  13   0  66  14   4]
 [  0   2 432  21   7  68   0]
 [  0   0  11 548   0  13   0]
 [  0  18   9   0 468   6   0]
 [  0   7  62  11   3 487   0]
 [ 25   1   0   0   0   0 532]]

               precision    recall  f1-score   support

           1       0.73      0.74      0.74       486
           2       0.78      0.64      0.70       563
           3       0.82      0.82      0.82       530
           4       0.94      0.96      0.95       572
           5       0.84      0.93      0.89       501
           6       0.83      0.85      0.84       570
           7       0.92      0.95      0.94       558

    accuracy                           0.84      3780
   macro avg       0.84      0.84      0.84      3780
weighted avg       0.84      0.84      0.84      3780


Process finished with exit code 0
```