import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle

n_cv_iter = 5

with open('X_y_feat_names.pkl', 'rb') as fo:
    X, y, feature_names = pickle.load(fo)
print("Data reading completed")

skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('scaling', StandardScaler()),
         ('classification',
          GradientBoostingClassifier(random_state=42, learning_rate=0.05,
                                     max_features='sqrt', subsample=0.8,
                                     n_estimators=70))]
          #GradientBoostingClassifier(verbose=0, n_estimators=100,
          #                           subsample=0.8, learning_rate=0.05,
          #                           max_depth=5, min_samples_leaf=5,
          #                           max_features=0.5)
pipeline = Pipeline(steps)


def modelfit(alg, X, y, feature_names, perfome_CV=True, print_FI=True, cv=None):
    alg.fit(X, y)

    train_pred = alg.predict(X)
    train_pred_prob = alg.predict_proba(X)[:, 1]

    if perfome_CV:
        cv_score = cross_val_score(alg, X, y, cv=cv, scoring='f1')
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score),
               np.max(cv_score)))
    if print_FI:
        feat_imp = pd.Series(alg.named_steps['classification'].feature_importances_,
                             feature_names).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title="Feature Importance")
        plt.ylabel("Feature Importance Score")
        plt.savefig("FI.png", bbox_inches='tight', dpi=300)


# modelfit(pipeline, X, y, feature_names, cv=skf)

from sklearn.grid_search import GridSearchCV

param_grid = {'classification__max_depth': (5, 8),
              'classification__min_samples_leaf': (5, 10)}

print("Searching best parameters...")
gs_cv = GridSearchCV(pipeline, param_grid, scoring='roc_auc', iid=False, cv=skf,
                     n_jobs=1).fit(X, y)
print("The best parameters are %s with a score of %0.7f"
      % (gs_cv.best_params_, gs_cv.best_score_))
print(gs_cv.grid_scores_)
feat_imp = pd.Series(gs_cv.best_estimator_.named_steps['classification'].feature_importances_,
                     feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', title="Feature Importance")
plt.ylabel("Feature Importance Score")
plt.savefig("FI_1.png", bbox_inches='tight', dpi=300)
