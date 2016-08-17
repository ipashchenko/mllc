import glob
import os
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from data_load import load_data


data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'

fnames = glob.glob(os.path.join(data_dir, '*.log'))[::-1]
n_cv_iter = 5
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID', 'Npts']
scores_dict = list()


X, y, feature_names = load_data(fnames, names, names_to_delete)
skf = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
skf_1 = StratifiedShuffleSplit(y, n_iter=1, test_size=1./n_cv_iter,
                               random_state=42)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('scaling', StandardScaler()),
         ('classification',
          GradientBoostingClassifier(verbose=0, n_estimators=800,
                                     subsample=0.6, learning_rate=0.01,
                                     max_depth=5, min_samples_leaf=20,
                                     max_features=7, min_samples_split=300))]
pipeline = Pipeline(steps)

# With calibration
calibrated_clf = CalibratedClassifierCV(pipeline, method='isotonic', cv=skf)
for train_index, test_index in skf_1:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
calibrated_clf.fit(X_train, y_train)
y_preds = calibrated_clf.predict_proba(X_test)
print "With calibration"
print "%.2f" % log_loss(y_test, y_preds, eps=1e-15, normalize=True)

# Without calibration
pipeline.fit(X_train, y_train)
y_preds = pipeline.predict_proba(X_test)
print "Without calibration"
print "%.2f" % log_loss(y_test, y_preds, eps=1e-15, normalize=True)

