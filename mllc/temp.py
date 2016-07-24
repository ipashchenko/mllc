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

# Convert to numpy arrays
# Features
X_0 = np.array(df_0[list(features_names)].values, dtype=float)
X_1 = np.array(df_1[list(features_names)].values, dtype=float)
# Responses
y_0 = np.zeros(len(X_0))
y_1 = np.ones(len(X_1))
y = np.hstack((y_0, y_1))

# Impute features (change NaN's to mean for that feature)
from sklearn.preprocessing import Imputer
imp_0 = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
imp_1 = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
imp_0.fit(X_0)
imp_1.fit(X_1)
X_0 = imp_0.transform(X_0)
X_1 = imp_1.transform(X_1)
X = np.vstack((X_0, X_1))

# Scale
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# Split data to train & test samples
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Try some models
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

model_rfc = RandomForestClassifier(n_estimators=100)
model_knc = KNeighborsClassifier(n_neighbors=10)
model_lr = LogisticRegression(penalty='l1', tol=0.01)
model_svc = svm.SVC()
models = {'RF': model_rfc, 'KN': model_knc, 'LR': model_lr, 'SV': model_svc}
kfold = 5

cv_results = dict()
for clf_name, model in models.items():
    print("Working with model {}".format(clf_name))
    scores = cross_validation.cross_val_score(model, X, y, cv=kfold)
    cv_results[clf_name] = scores.mean()

# Plot CV results
pd.DataFrame.from_dict(data=cv_results, orient='index').plot(kind='bar',
                                                             legend=False)

# Plot ROC curve
plt.clf()
plt.figure(figsize=(8, 6))

for clf_name, model in models.items():
    # For SVC model only
    try:
        model.probability = True
    except AttributeError:
        pass
    probas = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (clf_name, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
for clf_name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print clf_name, cm, precision(cm), recall(cm), f1(cm)
    np.set_printoptions(precision=2)
    sea.heatmap(cm_normalized)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


