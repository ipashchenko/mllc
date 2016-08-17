import glob
import os
from data_load import load_data
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.cross_validation import StratifiedShuffleSplit


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    seed = 1

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
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=1. / n_cv_iter,
                                 random_state=seed)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''

    model = Sequential()
    model.add(Dense(25, input_dim=25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(13, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    learning_rate = 0.05
    decay_rate = learning_rate / epochs
    momentum = 0.9
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])





    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        # We can also choose between complete sets of layers
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))