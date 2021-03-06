# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from data_load import load_data
from utils import print_cm_summary, cm_scores
from plotting import plot_importance
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import sklearn.pipeline

data_dir =\
    '/home/ilya/Dropbox/kirx/ogle_simulation_72iterations/indexes_normalized'
n_cv_iter = 5
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts']
scores_dict = dict()
X = list()
y = list()

for iteration in range(1, 73):
    print("Iteration {}".format(iteration))
    fnames = sorted(glob.glob(os.path.join(data_dir,
                                           "ITERATION0000{}*".format(str(iteration).zfill(2)))))
    X, y, feature_names = load_data(fnames, names, names_to_delete)

    skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
    imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
    steps = [('imputation', imp), ('scaling', StandardScaler()),
             ('classification',
              SVC(probability=True, class_weight='balanced', C=1., gamma=0.01))]
             #  GradientBoostingClassifier(verbose=0, n_estimators=3000,
             #                             subsample=0.8, learning_rate=0.01,
             #                             max_depth=5, min_samples_leaf=15,
             #                             max_features=0.5))]
    pipeline = sklearn.pipeline.Pipeline(steps)
    ps, rs, f1s = [], [], []
    for i, (train_index, test_index) in enumerate(skf):
        print("CV fold {} of {}".format(i + 1, n_cv_iter))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        clf = pipeline.named_steps['classification']
        # plot_importance(clf, feature_names,
        #                 'feature_importance_OGLE_sim{}_cv_{}.png'.format(str(iteration).zfill(2), i))
        cm = confusion_matrix(y_test, y_pred)
        try:
            print_cm_summary(cm)
            p, r, f1 = cm_scores(cm)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        except ZeroDivisionError:
            continue
    scores_dict.update({iteration: (np.mean(ps), np.mean(rs), np.mean(f1s))})

print scores_dict
