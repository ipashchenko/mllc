# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt


# Load data
data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'
file_1 = 'vast_lightcurve_statistics_normalized_variables_only.log'
file_0 = 'vast_lightcurve_statistics_normalized_constant_only.log'
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
df_1 = pd.read_table(os.path.join(data_dir, file_1), names=names,
                     engine='python', na_values='+inf', sep=r"\s*",
                     usecols=range(30))
df_0 = pd.read_table(os.path.join(data_dir, file_0), names=names,
                     engine='python', na_values='+inf', sep=r"\s*",
                     usecols=range(30))

# Remove meaningless features
for df in (df_0, df_1):
    del df['Magnitude']
    del df['meaningless_1']
    del df['meaningless_2']
    del df['star_ID']
    df['CSSD'] = np.log(df['CSSD'] + df['CSSD'].median())

# List of feature names
features_names = list(df_0)
# Count number of NaN for eac feature
for feature in features_names:
    print(feature, df_0[feature].isnull().sum())
for feature in features_names:
    print(feature, df_1[feature].isnull().sum())

# Plot correlation matrix
for df in (df_0, df_1):
    corr_matrix = df.corr()
    from plotting import plot_corr_matrix
    plot_corr_matrix(corr_matrix)
plt.close()

# Convert to numpy arrays
# Features
X_0 = np.array(df_0[list(features_names)].values, dtype=float)
X_1 = np.array(df_1[list(features_names)].values, dtype=float)
X = np.vstack((X_0, X_1))
# Responses
y_0 = np.zeros(len(X_0))
y_1 = np.ones(len(X_1))
y = np.hstack((y_0, y_1))

# Split data to train & test samples and fit scaler & imputer on training sample
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Train data: ", len(y_train),
      float(np.count_nonzero(y_train)) / len(y_train))
print("Test data: ", len(y_test),
      float(np.count_nonzero(y_test)) / len(y_test))

# Fit scaler & imputer on training data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
imp.fit(X_train)
scaler = StandardScaler().fit(imp.transform(X_train))
X_trained_scaled = scaler.transform(imp.transform(X_train))
# Use the same transformation & imputation for testing data!
X_test_scaled = scaler.transform(imp.transform(X_test))

# Try some models
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline

# Just use arbitrary values of hyper parameters
model_rf = RandomForestClassifier(n_estimators=100)
model_kn = KNeighborsClassifier(n_neighbors=10)
model_lr = LogisticRegression(penalty='l1', tol=0.01)
model_sv = svm.SVC()
models = {'RF': model_rf, 'KN': model_kn, 'LR': model_lr, 'SV': model_sv}
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
pipes = {clf_name: make_pipeline(imp, StandardScaler(), model) for
         clf_name, model in models.items()}

cv_results = dict()
from sklearn.cross_validation import StratifiedShuffleSplit
n_cv_iter = 5
skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
for clf_name, pipe in pipes.items():
    print("Working with model {}".format(clf_name))
    scores = cross_validation.cross_val_score(pipe, X, y, cv=skf,
                                              scoring='f1_weighted')
    print("CV scores: ", scores)
    cv_results[clf_name] = scores.mean()

# Plot CV results
fig, ax = plt.subplots(figsize=(10, 10))
title = "{}-fold CV score".format(n_cv_iter)
pd.DataFrame.from_dict(data=cv_results, orient='index').plot(kind='bar',
                                                             legend=False,
                                                             ax=ax,
                                                             title=title)
fig.savefig("CV_scores.png", bbox_inches='tight', dpi=500)
plt.close()

# Plot ROC curve
plt.clf()
plt.figure(figsize=(8, 6))

for clf_name, model in models.items():
    # For SVC model only
    try:
        model.probability = True
    except AttributeError:
        pass
    probas = model.fit(X_trained_scaled, y_train).predict_proba(X_test_scaled)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (clf_name, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.savefig("ROC.png", bbox_inches='tight', dpi=500)
plt.close()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
from utils import print_cm_summary
for clf_name, model in models.items():
    y_pred = model.fit(X_trained_scaled, y_train).predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(clf_name)
    print_cm_summary(cm)
    np.set_printoptions(precision=2)
    sea.heatmap(cm_normalized)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(clf_name)
    plt.savefig("confusion_matrix_{}.png".format(clf_name), bbox_inches='tight',
                dpi=500)
    plt.close()

# For each CV-fold print CM
pipe = pipes['RF']
skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
for i, (train_index, test_index) in enumerate(skf):
    print("CV fold {} of {}".format(i + 1, n_cv_iter))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train data: ", len(y_train),
          float(np.count_nonzero(y_train)) / len(y_train))
    print("Test data: ", len(y_test),
          float(np.count_nonzero(y_test)) / len(y_test))
    y_pred = pipe.fit(X_train, y_train).predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print_cm_summary(cm)


