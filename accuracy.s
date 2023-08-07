import numpy as np import
pandas as pd
import matplotlib.pyplot as plt #data visualizationimport seaborn as sns
#data visualization
from sklearn.metrics import accuracy_score from sklearn.metrics
import confusion_matrix
from sklearn.preprocessing import StandardScaler from
sklearn.model_selection import train_test_splitfrom sklearn.ensemble
import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer(as_frame=True)df=dataset.frame
#dividing data
X = df.drop("target",axis = 1)y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42,shuffle=True,
stratify=y)
accuracy=[]
nodes=[]
for i in range(1,200):
#Create a Gaussian Classifier RFclfModel=RandomForestClassifier(n_estimators=i)
#Train the model using the training sets y_pred=clf.predict(X_test)RFclfModel.fit(X_train,y_train)
y_pred_RFclfModel=RFclfModel.predict(X_test)
# Model Accuracy, how often is the classifier correct? acc =
metrics.accuracy_score(y_test, y_pred_RFclfModel)accuracy.append(acc)
nodes.append(i)
plt.plot(nodes,accuracy) plt.xlabel("Number of nodes")
plt.ylabel("Accuracy achieved")
plt.title("Variation of accuracy with number of nodes")plt.show()