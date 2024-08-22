import json
import sys
import sklearn.model_selection as skm
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score,plot_roc_curve, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler 
from sklearn import tree
from matplotlib import pyplot as plt
from mlxtend.evaluate import mcnemar, mcnemar_table

data = pd.read_csv(f'data/processed/data_3it.csv')

input_names = list(data.columns)
input_names.remove('status')

all_labels = data['status'].values
all_inputs = data[input_names].values

(training_inputs,
 testing_inputs,
 training_labels,
 testing_labels) = skm.train_test_split(all_inputs, all_labels, test_size=0.20, random_state=1, stratify=all_labels)

scaler = StandardScaler()

scaler.fit(training_inputs)
training_inputs = scaler.transform(training_inputs)
testing_inputs = scaler.transform(testing_inputs) 

over_sampler = SMOTE()
inputs, labels = over_sampler.fit_resample(training_inputs, training_labels)

testing_labels_prob = [1 if x==-1 else 0 for x in testing_labels]

rf = RandomForestClassifier(max_depth=None, max_features="log2", min_samples_leaf=1, min_samples_split=2, n_estimators=200)

rf.fit(inputs, labels)
rf_prediction_prob = rf.predict_proba(testing_inputs)
rf_prediction = rf.predict(testing_inputs)
rf_prediction_prob = [x[0] for x in rf_prediction_prob]
rf_best_auc = roc_auc_score(testing_labels_prob, rf_prediction_prob)

RocCurveDisplay.from_predictions(testing_labels_prob, rf_prediction_prob)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve for Random Forest evaluation")
plt.savefig("graphs/roc_rf.pdf")


lr = LogisticRegression(C=1, dual=False, penalty="l2", tol=0.001)

lr.fit(inputs, labels)
lr_prediction_prob = lr.predict_proba(testing_inputs)
lr_prediction = lr.predict(testing_inputs)
lr_prediction_prob = [x[0] for x in lr_prediction_prob]
lr_best_auc = roc_auc_score(testing_labels_prob, lr_prediction_prob)

RocCurveDisplay.from_predictions(testing_labels_prob, lr_prediction_prob)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve for Logistic Regression evaluation")
plt.savefig("graphs/roc_lr.pdf")

tb = mcnemar_table(y_target=testing_labels, 
                   y_model1=lr_prediction, 
                   y_model2=rf_prediction)

chi2, p = mcnemar(ary=tb, corrected=True)

with open("./McNemar.txt", "w") as f:
    f.write("auc rf: "+ str(rf_best_auc))
    f.write("auc_lr: "+ str(lr_best_auc))
    f.write("p-value: " + str(p))
    f.write("chi_squared: " + str(chi2))