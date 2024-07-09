# Basic packages always been used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Data preprocessing useful function
from sklearn.preprocessing import LabelEncoder

# Function for spilting training & testing data set
from sklearn.model_selection import train_test_split, GridSearchCV

# Algorithm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Functions for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,roc_curve,auc, RocCurveDisplay 
from imblearn.metrics import sensitivity_score, specificity_score

file_path = r"C:\Users\tiffa\Predict-diabetes\diabetes.csv"
df = pd.read_csv(file_path)

#data preprocessing(清除錯誤資料)
condition = ((df['Glucose'] == 0) | (df['BloodPressure'] == 0) | (df['SkinThickness'] == 0) | (df['BMI'] == 0))
df_filtered = df[~condition]

# Export to CSV file
df_filtered.to_csv(r"C:\Users\tiffa\Predict-diabetes\diabetes_deletezero.csv",index=False, header=True)
print('Export complete...')

#load data(sometimes need to use "/")
dummied_new_df = pd.read_csv(r"C:\Users\tiffa\Predict-diabetes\diabetes_deletezero.csv")

#set x (make prediction), y (want to predict)
x=dummied_new_df.drop(['Outcome'],axis=1).copy()
y=dummied_new_df['Outcome'].copy()

#Split the data (split into 80% training data & 20% testing data) (lock seed)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11) #test_size的意思是把80%資料做trainning，20%的資料做testing

# Fit the model with DecisionTreeClassifier function
model=DecisionTreeClassifier(criterion='gini',max_depth=None,max_leaf_nodes=None,min_samples_leaf=1,random_state=11)
model.fit(x_train,y_train)

# Perform  five fold cross validation on training data
CV5F_acc=cross_val_score(model,x_train,y_train,cv=5,scoring='accuracy')
print('Each fold ACC:',CV5F_acc)
print('Average ACC:',round((np.mean(CV5F_acc))*100,2),'+/-',round((np.std(CV5F_acc))*100,2))


#hyper parameter set
param_grid = {
    'max_depth': list(range(1, 11)),
    'max_leaf_nodes': list(range(2, 11)),
    'min_samples_leaf': list(range(1, 21))
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=4),
                           param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise')
grid_search.fit(x_train, y_train)

print("Best parameters: ", grid_search.best_params_)


# Evaluate with testing data set
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# Evaluate with testing data set
final_model_testing_prediction = best_model.predict(x_test)
final_model_testing_acc = accuracy_score(y_test, final_model_testing_prediction)
final_model_testing_f1s = f1_score(y_test, final_model_testing_prediction)
final_model_testing_pre = precision_score(y_test, final_model_testing_prediction)
final_model_testing_sen = recall_score(y_test, final_model_testing_prediction)
final_model_testing_spe = recall_score(y_test, final_model_testing_prediction, pos_label=0)
final_model_testing_cm = confusion_matrix(y_test, final_model_testing_prediction)

# Print results
print('Testing ACC:', round(final_model_testing_acc * 100, 2))
print('Testing f1s:', round(final_model_testing_f1s * 100, 2))
print('Testing pre:', round(final_model_testing_pre * 100, 2))
print('Testing sen:', round(final_model_testing_sen * 100, 2))
print('Testing spe:', round(final_model_testing_spe * 100, 2))

# confusion matrix
ConfusionMatrixDisplay(final_model_testing_cm, display_labels=["0 no diabetes", "1 have diabetes"]).plot()

# decision tree
plt.figure(figsize=(15, 7.5))
tree.plot_tree(best_model, filled=True, rounded=True, class_names=["not survived", "survived"], feature_names=x.columns)
plt.show()