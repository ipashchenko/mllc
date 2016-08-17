# -*- coding: utf-8 -*-
import glob
import os
from data_load import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

n_cv_iter = 5
sim_data_dir = \
    '/home/ilya/Dropbox/kirx/ogle_simulation_72iterations/indexes_normalized'
data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts']

fnames_var = sorted(glob.glob(os.path.join(sim_data_dir,
                                           "ITERATION0000*variable*")))[:30]
fname_const = glob.glob(os.path.join(data_dir, '*.log'))[::-1][0]
fnames_var.insert(0, fname_const)
fnames = fnames_var

X, y, feature_names = load_data(fnames, names, names_to_delete)
print feature_names

skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('scaling', StandardScaler()),
         ('classification',
          GradientBoostingClassifier(random_state=42, learning_rate=0.01,
                                     max_features=7,
                                     n_estimators=800,
                                     subsample=0.6,
                                     min_samples_split=300,
                                     max_depth=7))]

pipeline = Pipeline(steps)


def modelfit(alg, X, y, feature_names, perfome_CV=True, print_FI=True, cv=None):
    alg.fit(X, y)

    train_pred = alg.predict(X)
    train_pred_prob = alg.predict_proba(X)[:, 1]

    if perfome_CV:
        cv_score = cross_val_score(alg, X, y, cv=cv, scoring='roc_auc')
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score),
               np.max(cv_score)))
    if print_FI:
        feat_imp = pd.Series(alg.named_steps['classification'].feature_importances_,
                             feature_names).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title="Feature Importance")
        plt.ylabel("Feature Importance Score")
        plt.show()
        # plt.savefig("FI.png", bbox_inches='tight', dpi=300)


# modelfit(pipeline, X, y, feature_names, cv=skf)

skf = StratifiedShuffleSplit(y, n_iter=1, test_size=0.2)
for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_score = pipeline.predict_proba(X_test)
y_true = y_test


# from sklearn.grid_search import GridSearchCV
#
# param_grid = {'classification__subsample': [0.5, 0.6,0.7,0.75,0.8,0.85,0.9]}
#
# print("Searching best parameters...")
# gs_cv = GridSearchCV(pipeline, param_grid, scoring='roc_auc', iid=False, cv=skf,
#                      n_jobs=2).fit(X, y)
# print("The best parameters are %s with a score of %0.7f"
#       % (gs_cv.best_params_, gs_cv.best_score_))
# gs_cv.grid_scores_
# feat_imp = pd.Series(gs_cv.best_estimator_.named_steps['classification'].feature_importances_,
#                      feature_names).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title="Feature Importance")
# plt.ylabel("Feature Importance Score")
# plt.show()
