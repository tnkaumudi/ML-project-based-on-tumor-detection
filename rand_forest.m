import numpy as np import
pandas as pd
import matplotlib.pyplot as plt #data visualizationimport seaborn as sns
#data visualization
from sklearn.metrics import accuracy_score from sklearn.metrics
import confusion_matrix
from sklearn.preprocessing import StandardScaler from
sklearn.model_selection import train_test_splitfrom sklearn.ensemble
import RandomForestClassifier from sklearn import metrics
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer(as_frame=True)df=dataset.frame
#division of data
X = df.drop("target",axis = 1)y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42,shuffle=True,
stratify=y)
#Creation of a Gaussian Classifier RFclfModel=RandomForestClassifier(n_estimators=100)
#Model is trained using the training sets y_pred=clf.predict(X_test)RFclfModel.fit(X_train,y_train)
y_pred_RFclfModel=RFclfModel.predict(X_test)
# Accuracy of model
print("Accuracy:",metrics.accuracy_score(y_test,y_pred_RFclfModel))
#confusion_matrix
cv_randForest = confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred_RFclfModel, rownames=['True'], colnames=['Predicted'], margins=True)
sns.heatmap(cv_randForest,annot=True)
