import json
import sys
import sklearn.model_selection as skm
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score,plot_roc_curve
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler 
from sklearn import tree
from matplotlib import pyplot as plt

iteration = int(sys.argv[1])
print(iteration)

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(f'data/processed/data_{iteration}it.csv')

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

# rf_param_grid = {
#     'max_depth': [10, 20, 30, None],
#     'max_features': ['sqrt', 'log2', None],
#     'min_samples_leaf': [1, 2, 4, 6, 10],
#     'min_samples_split': [2, 5, 10, 20, 40],
#     'n_estimators': [50, 75, 100, 200]}

# rf_grid_search = GridSearchCV(RandomForestClassifier(),
#                            param_grid=rf_param_grid,
#                            cv=10,
#                            scoring='roc_auc',
#                            n_jobs=-1)

# rf_grid_search.fit(inputs, labels)
# print('Best score: {}'.format(rf_grid_search.best_score_))
# print('Best parameters: {}'.format(rf_grid_search.best_params_))
# print('Best estimator: {}'.format(rf_grid_search.best_estimator_))

# rf = rf_grid_search.best_estimator_
# rf_prediction = rf.predict_proba(testing_inputs)
# rf_prediction = [x[0] for x in rf_prediction]
# best_auc = roc_auc_score(testing_labels_prob, rf_prediction)

# with open(f"./reports/predictive/rf_report_{iteration}it.json", "w") as outfile:
#     json.dump([{'auc': best_auc}, {'best_params': rf_grid_search.best_params_}], outfile)

# knn_param_grid = {'n_neighbors': [4, 5, 6, 7, 8, 9, 10],
#                   'weights': ['uniform', 'distance'],
#                   'algorithm': ['ball_tree', 'kd_tree', 'brute']}

# knn_grid_search = GridSearchCV(KNeighborsClassifier(),
#                            param_grid=knn_param_grid,
#                            cv=10,
#                            scoring='roc_auc',
#                            n_jobs=-1)


# knn_grid_search.fit(inputs, labels)
# print('Best score: {}'.format(knn_grid_search.best_score_))
# print('Best parameters: {}'.format(knn_grid_search.best_params_))
# print('Best estimator: {}'.format(knn_grid_search.best_estimator_))

# knn = knn_grid_search.best_estimator_
# knn_prediction = knn.predict(testing_inputs)
# best_auc = roc_auc_score(testing_labels, knn_prediction)

# with open(f"./reports/predictive/knn_report_{iteration}it.json", "w") as outfile:
#     json.dump([{'auc': best_auc}, {'best_params': knn_grid_search.best_params_}], outfile)

# svc_param_grid = {'C' : [0.1, 1, 10, 100], 
#             'gamma' : ['scale', 'auto'],
#             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#             'probability': [True]}

# svc_grid_search = GridSearchCV(SVC(),
#                            param_grid=svc_param_grid,
#                            cv=10,
#                            scoring='roc_auc',
#                            n_jobs=-1)


# svc_grid_search.fit(inputs, labels)
# print('Best score: {}'.format(svc_grid_search.best_score_))
# print('Best parameters: {}'.format(svc_grid_search.best_params_))
# print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

# svc = svc_grid_search.best_estimator_
# svc_prediction = svc.predict_proba(testing_inputs)
# svc_prediction = [x[0] for x in svc_prediction]
# best_auc = roc_auc_score(testing_labels_prob, svc_prediction)
# print(svc_grid_search.cv_results_)

# with open(f"./reports/predictive/svc_report_{iteration}it.json", "w") as outfile:
#     json.dump([{'auc': best_auc}, {'best_params': svc_grid_search.best_params_}], outfile)

dtc_param_grid = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [2,4, 6 ,8, 10, 12, None],
                   'max_features': [12, 13, 14, 15, 16],
                   'class_weight' : [None, {'1':1, '-1':2}]}

dtc_grid_search = GridSearchCV(DecisionTreeClassifier(),
                           param_grid=dtc_param_grid,
                           cv=10,
                           scoring='roc_auc',
                           n_jobs=-1)


dtc_grid_search.fit(inputs, labels)
print('Best score: {}'.format(dtc_grid_search.best_score_))
print('Best parameters: {}'.format(dtc_grid_search.best_params_))
print('Best estimator: {}'.format(dtc_grid_search.best_estimator_))

dtc = dtc_grid_search.best_estimator_
dtc_prediction = dtc.predict(testing_inputs)
best_auc = roc_auc_score(testing_labels, dtc_prediction)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc, feature_names=input_names, class_names=["-1","1"])
fig.savefig("graphs/decistion_tree.pdf")

with open(f"./reports/predictive/dtc_report_{iteration}it.json", "w") as outfile:
    json.dump([{'auc': best_auc}, {'best_params': dtc_grid_search.best_params_}], outfile)


# lr_param_grid = {'C' : [0.1, 1, 10, 100], 
#             'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
#             'dual' : [True, False],
#             'tol': [1e-3,1e-4,1e-5],
#             }

# lr_grid_search = GridSearchCV(LogisticRegression(),
#                            param_grid=lr_param_grid,
#                            cv=10,
#                            scoring='roc_auc',
#                            n_jobs=-1)


# lr_grid_search.fit(inputs, labels)
# print('Best score: {}'.format(lr_grid_search.best_score_))
# print('Best parameters: {}'.format(lr_grid_search.best_params_))
# print('Best estimator: {}'.format(lr_grid_search.best_estimator_))

# lr = lr_grid_search.best_estimator_
# lr_prediction = lr.predict_proba(testing_inputs)
# lr_prediction = [x[0] for x in lr_prediction]
# best_auc = roc_auc_score(testing_labels_prob, lr_prediction)

# with open(f"./reports/predictive/lr_report_{iteration}it.json", "w") as outfile:
#     json.dump([{'auc': best_auc}, {'best_params': lr_grid_search.best_params_}], outfile)