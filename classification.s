#classification report
report = classification_report(y_test,y_pred_RFclfModel,labels=[1,0],ta
rget_names=["Malignant","Benign"])
print(report)