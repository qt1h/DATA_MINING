# Output

```Aperçu des données:
         obj_ID       alpha      delta         u         g         r         i  ...  field_ID   spec_obj_ID   class  redshift  plate    MJD fiber_ID
0  1.237661e+18  135.689107  32.494632  23.87882  22.27530  20.39501  19.16573  ...        79  6.543777e+18  GALAXY  0.634794   5812  56354      171
1  1.237665e+18  144.826101  31.274185  24.77759  22.83188  22.58444  21.16812  ...       119  1.176014e+19  GALAXY  0.779136  10445  58158      427
2  1.237661e+18  142.188790  35.582444  25.26307  22.66389  20.60976  19.34857  ...       120  5.152200e+18  GALAXY  0.644195   4576  55592      299
3  1.237663e+18  338.741038  -0.402828  22.13682  23.77656  21.61162  20.50454  ...       214  1.030107e+19  GALAXY  0.932346   9149  58039      775
4  1.237680e+18  345.282593  21.183866  19.43718  17.58028  16.49747  15.97711  ...       137  6.891865e+18  GALAXY  0.116123   6121  56187      842

[5 rows x 18 columns]

Informations sur le dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 18 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   obj_ID       100000 non-null  float64
 1   alpha        100000 non-null  float64
 2   delta        100000 non-null  float64
 3   u            100000 non-null  float64
 4   g            100000 non-null  float64
 5   r            100000 non-null  float64
 6   i            100000 non-null  float64
 7   z            100000 non-null  float64
 8   run_ID       100000 non-null  int64  
 9   rerun_ID     100000 non-null  int64  
 10  cam_col      100000 non-null  int64  
 11  field_ID     100000 non-null  int64  
 12  spec_obj_ID  100000 non-null  float64
 13  class        100000 non-null  object 
 14  redshift     100000 non-null  float64
 15  plate        100000 non-null  int64  
 16  MJD          100000 non-null  int64  
 17  fiber_ID     100000 non-null  int64  
dtypes: float64(10), int64(7), object(1)
memory usage: 13.7+ MB
None

Valeurs manquantes par colonne:
obj_ID         0
alpha          0
delta          0
u              0
g              0
r              0
i              0
z              0
run_ID         0
rerun_ID       0
cam_col        0
field_ID       0
spec_obj_ID    0
class          0
redshift       0
plate          0
MJD            0
fiber_ID       0
dtype: int64

K-Nearest Neighbors - Accuracy: 0.8996
Classification Report:
               precision    recall  f1-score   support

      GALAXY       0.88      0.97      0.92     11860
         QSO       0.97      0.81      0.88      3797
        STAR       0.91      0.79      0.85      4343

    accuracy                           0.90     20000
   macro avg       0.92      0.86      0.88     20000
weighted avg       0.90      0.90      0.90     20000

Matrice de confusion enregistrée dans : confusion_matrices/confusion_matrix_K-Nearest_Neighbors.png

Decision Tree - Accuracy: 0.9640
Classification Report:
               precision    recall  f1-score   support

      GALAXY       0.97      0.97      0.97     11860
         QSO       0.91      0.91      0.91      3797
        STAR       1.00      0.99      0.99      4343

    accuracy                           0.96     20000
   macro avg       0.96      0.96      0.96     20000
weighted avg       0.96      0.96      0.96     20000

Matrice de confusion enregistrée dans : confusion_matrices/confusion_matrix_Decision_Tree.png

Naïve Bayes - Accuracy: 0.7088
Classification Report:
               precision    recall  f1-score   support

      GALAXY       0.75      0.86      0.80     11860
         QSO       0.58      0.88      0.70      3797
        STAR       0.98      0.16      0.27      4343

    accuracy                           0.71     20000
   macro avg       0.77      0.63      0.59     20000
weighted avg       0.77      0.71      0.67     20000

Matrice de confusion enregistrée dans : confusion_matrices/confusion_matrix_Naïve_Bayes.png

Support Vector Machine - Accuracy: 0.9614
Classification Report:
               precision    recall  f1-score   support

      GALAXY       0.96      0.97      0.97     11860
         QSO       0.95      0.89      0.92      3797
        STAR       0.96      1.00      0.98      4343

    accuracy                           0.96     20000
   macro avg       0.96      0.95      0.96     20000
weighted avg       0.96      0.96      0.96     20000

Matrice de confusion enregistrée dans : confusion_matrices/confusion_matrix_Support_Vector_Machine.png

Multi-Layer Perceptron - Accuracy: 0.9682
Classification Report:
               precision    recall  f1-score   support

      GALAXY       0.98      0.97      0.97     11860
         QSO       0.96      0.93      0.94      3797
        STAR       0.95      1.00      0.98      4343

    accuracy                           0.97     20000
   macro avg       0.96      0.97      0.96     20000
weighted avg       0.97      0.97      0.97     20000

Matrice de confusion enregistrée dans : confusion_matrices/confusion_matrix_Multi-Layer_Perceptron.png
...
```
