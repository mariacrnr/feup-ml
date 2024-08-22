import numpy as np
import sklearn.model_selection as skm
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import sys

iteration = int(sys.argv[1])
print(iteration)

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(f'data/processed/data_{iteration}it.csv')

input_names = list(data.columns)
input_names.remove('status')

all_labels = data['status'].values
all_inputs = data[input_names].values

(X,
 testing_inputs,
 y,
 testing_labels) = skm.train_test_split(all_inputs, all_labels, test_size=0.20, random_state=1, stratify=all_labels)

scaler = StandardScaler()

scaler.fit(X)
training_inputs = scaler.transform(X)
testing_inputs = scaler.transform(testing_inputs) 

over_sampler = SMOTE()
inputs, labels = over_sampler.fit_resample(training_inputs, y)

testing_labels_prob = [1 if x==-1 else 0 for x in testing_labels]

models = {'lr' : LogisticRegression(C=1, dual=False, penalty="l2", tol=0.001),
         'svc' : SVC(C=10, gamma="scale", kernel="rbf", probability=True),
         'rf' : RandomForestClassifier(max_depth=None, max_features="log2", min_samples_leaf=1, min_samples_split=2, n_estimators=200),
         'knn' : KNeighborsClassifier(algorithm="ball_tree", n_neighbors=10, weights="distance"),
         'dtc' :  DecisionTreeClassifier(class_weight=None, criterion="gini", max_depth=8, max_features=15, splitter="random")
    }

colors = {'lr' : ["yellow","khaki"], 'svc' : ["blue", "lavender"], 'rf' : ["orange", "moccasin"], "knn" : ["green", "palegreen"], "dtc" : ["violet", "thistle"]}

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

cv = StratifiedKFold(n_splits=10,shuffle=False)
for model in models:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig1, ax1 = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        models[model].fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            models[model],
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax1,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colors[model][0],
        label= model + r" - Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color=colors[model][1],
        alpha=0.2,
    )

    ax.set(
        xlim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylim=[-0.05, 1.05],
        ylabel="True Positive Rate",
        title="Validation scores for tested models",
    )
    ax.legend(loc="lower right")
fig.savefig(f"graphs/it{iteration}.pdf")
