# -*- coding: utf-8 -*-
import glob
import os

from sklearn_evaluation.plot import confusion_matrix as plot_cm

from data_load import load_data
import numpy
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix

from utils import print_cm_summary
import matplotlib.pyplot as plt

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

# load dataset
data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'

fnames = glob.glob(os.path.join(data_dir, '*.log'))[::-1]
n_cv_iter = 5
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts']
X, y, feature_names = load_data(fnames, names, names_to_delete)


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=25, init='normal', activation='relu'))
                    # W_constraint=maxnorm(3)))
    model.add(Dense(25, init='normal', activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(13, init='normal', activation='relu'))
                    # W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


epochs = 60
batch_size = 10
# estimators = list()
# estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
#                                       axis=0, verbose=2)))
# estimators.append(('scaler', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
#                                           nb_epoch=epochs,
#                                           batch_size=batch_size,
#                                           verbose=1)))
# estimators.append(('classification',
#                    GradientBoostingClassifier(random_state=42,
#                                               learning_rate=0.005,
#                                               max_features=7,
#                                               n_estimators=1600,
#                                               subsample=0.6,
#                                               min_samples_split=300,
#                                               max_depth=7)))
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=1. / n_cv_iter,
                             random_state=seed)
skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
# imp.fit(X)
imp.fit(X_train)
scaler = StandardScaler().fit(imp.transform(X_train))
# scaler = StandardScaler().fit(imp.transform(X))
X_trained_scaled = scaler.transform(imp.transform(X_train))
# # Use the same transformation & imputation for testing data!
X_test_scaled = scaler.transform(imp.transform(X_test))
X_scaled = scaler.transform(imp.transform(X))

# pipeline = Pipeline(estimators)

model = create_baseline()

# history = model.fit(X_scaled, y, callbacks=[remote], batch_size=batch_size,
#                     nb_epoch=epochs)
history = model.fit(X_trained_scaled, y_train,
                    validation_data=(X_test_scaled, y_test),
                    batch_size=batch_size, nb_epoch=epochs)
                    # class_weight={0: 1., 1: 10.})
# results = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=2)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = model.predict(X_test_scaled)
y_pred[y_pred < 0.5] = 0.
y_pred[y_pred >= 0.5] = 1.
y_probs = model.predict_proba(X_test_scaled)
# plot_cm(y_test, y_pred, target_names=['const', 'var'])
cm = confusion_matrix(y_test, y_pred)
print_cm_summary(cm)
#
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()