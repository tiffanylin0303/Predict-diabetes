# Basic packages always been used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,roc_curve,auc, RocCurveDisplay
from imblearn.metrics import sensitivity_score, specificity_score

file_path = r"C:\Users\tiffa\Predict-diabetes\diabetes_deletezero.csv"
df = pd.read_csv(file_path)

#set x (make prediction) with minimax 
x=df.drop(['Outcome'],axis=1).copy()

#minimax scaling  把數字壓到零跟一之間，算是一種資料前處理
MMscaler=MinMaxScaler(feature_range=(0, 1))
scaling=MMscaler.fit_transform(x)
scaled_data=pd.DataFrame(data=scaling)
scaled_data.columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

#set y (want to predict)
y=df['Outcome'].copy()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)

#hyper parameter set
param_grid = {
    'degree': list(range(2, 10)),
    'C': list(range(1, 10)),
    'kernel' : ["poly"]
}

grid_search = GridSearchCV(estimator=SVC(random_state=11), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best parameters: ", grid_search.best_params_)


# Evaluate with testing data set
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)


# Evaluate with testing data set
final_model_testing_prediction = best_model.predict(x_test)
final_model_testing_acc=accuracy_score(y_test,final_model_testing_prediction)
final_model_testing_f1s=f1_score(y_test,final_model_testing_prediction,pos_label=1)
final_model_testing_pre=precision_score(y_test,final_model_testing_prediction,pos_label=1)
final_model_testing_sen=sensitivity_score(y_test,final_model_testing_prediction,pos_label=1)
final_model_testing_spe=specificity_score(y_test,final_model_testing_prediction,pos_label=1)
final_model_testing_cm=confusion_matrix(y_test,final_model_testing_prediction)


print('Testing ACC:',round(final_model_testing_acc*100,2))
print('Testing f1s:',round(final_model_testing_f1s*100,2))
print('Testing pre:',round(final_model_testing_pre*100,2))
print('Testing sen:',round(final_model_testing_sen*100,2))
print('Testing spe:',round(final_model_testing_spe*100,2))
print(ConfusionMatrixDisplay(final_model_testing_cm, display_labels=["0 no diabetes", "1 have diabetes"]).plot())