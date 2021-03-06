import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from data_load import load_data


import os
import glob
data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'

fnames = glob.glob(os.path.join(data_dir, '*.log'))[::-1]
n_cv_iter = 5
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp',
         'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS',
         'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts']

X, y, feature_names = load_data(fnames, names, names_to_delete)

skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('scaling', StandardScaler()),
         ('classification',
          KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=1))]
          # SVC(probability=True, class_weight='balanced', C=1., gamma=0.01))]
          # GradientBoostingClassifier(random_state=42, learning_rate=0.0025,
          #                            n_estimators=1200, max_depth=5,
          #                            min_samples_leaf=4, min_samples_split=50,
          #                            max_features=7, subsample=0.7))]
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


# modelfit(pipeline, X, y, feature_names, cv=skf, print_FI=False)

from sklearn.grid_search import GridSearchCV

param_grid = {'classification__n_neighbors': [250, 300, 350],
              'classification__weights': ['distance']}

print("Searching best parameters...")
gs_cv = GridSearchCV(pipeline, param_grid, scoring='roc_auc', iid=False, cv=skf,
                     n_jobs=4).fit(X, y)
print("The best parameters are %s with a score of %0.7f"
      % (gs_cv.best_params_, gs_cv.best_score_))
print gs_cv.grid_scores_
# feat_imp = pd.Series(gs_cv.best_estimator_.named_steps['classification'].feature_importances_,
#                      feature_names).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title="Feature Importance")
# plt.ylabel("Feature Importance Score")
# plt.show()
# plt.savefig("FI_0.png", bbox_inches='tight', dpi=300)
