import numpy as np
import sklearn.model_selection as skm
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import RocCurveDisplay

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed/data_5it.csv')
test = pd.read_csv('data/processed/comp_5it.csv')

input_names = list(train.columns)
input_names.remove('status')

testing_inputs = test[input_names]

training_labels = train['status']
training_inputs = train[input_names]

scaler = StandardScaler()

testing_ids = testing_inputs['loan_id'].values

training_inputs = scaler.fit_transform(training_inputs)
testing_inputs = scaler.transform(testing_inputs) 


over_sampler = SMOTE()
inputs, labels = over_sampler.fit_resample(training_inputs, training_labels)

param_grid = {
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_leaf': [1, 2, 4, 6, 10],
    'min_samples_split': [2, 5, 10, 20, 40],
    'n_estimators': [50, 75, 100, 200]}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid,
                           cv=10,
                           scoring='roc_auc',
                           n_jobs=-1)

# param_grid = {'C' : [0.1, 1, 10, 100], 
#             'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
#             'dual' : [True, False],
#             'tol': [1e-3,1e-4,1e-5],
#             }

# grid_search = GridSearchCV(LogisticRegression(),
#                            param_grid=param_grid,
#                            cv=6,
#                            scoring='roc_auc',
#                            n_jobs=-1)



grid_search.fit(inputs, labels)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Best estimator: {}'.format(grid_search.best_estimator_))

clf = grid_search.best_estimator_
clf_prediction = clf.predict_proba(testing_inputs)

sub_data = {'Id': testing_ids, 'Predicted': [x[0] for x in clf_prediction]}
sub = pd.DataFrame(data=sub_data)

csv = sub.to_csv(index = False)
with open("./submission/it9.csv", "w") as outfile:
    outfile.write(csv)
