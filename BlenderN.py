import time
import keras
import numpy as np
import pandas as pd
from copy import copy
from sklearn.model_selection import KFold, StratifiedKFold



class Blender(object):
    def __init__(self, clfs = None, ensemble=None, stratify=False, original=False, epoch=10, batch_size=128):
        self.clfs = clfs
        self.ensemble = ensemble
        self.X_holdoutEnsemble = None
        self.X_testEnsemble = None
        self.base_models = []
        self.stratify = stratify
        self.kf = None
        self.original = original
        self.epoch = epoch
        self.batch_size = batch_size

    def get_pred(self, clf, X):
        if hasattr(clf, 'predict_proba'):
            if type(clf) == keras.models.Sequential:
                pred = clf.predict_proba(X, verbose=0)
            else:
                pred = clf.predict_proba(X)
        else:
            pred = clf.predict(X)

        return pred

    def fit_base(self, X, y):
        print ('Fitting Base Level Models...')

        for j, clf in enumerate(self.clfs):
            if type(clf) == keras.models.Sequential:
                clf_name = "Model %d: %s" %(j+1, clf)
                print('(%s) Fitting %s' % (time.asctime(time.localtime(time.time())), clf_name))
                clf.fit(X, pd.get_dummies(y).as_matrix(), nb_epoch=self.epoch, batch_size=self.batch_size, verbose=0)
                self.base_models.append(copy(clf))
            else:
                clf_name = "Model %d: %s" % (j + 1, clf)
                print('(%s) Fitting %s' % (time.asctime(time.localtime(time.time())), clf_name))
                clf.fit(X, y)
                self.base_models.append(copy(clf))

    def pred_base(self, X):
        X_trainEnsemble = None
        for j, clf in enumerate(self.base_models):

            X_trainEnsemble_clf = self.get_pred(clf, X)
            classes = X_trainEnsemble_clf.shape[1]

            if X_trainEnsemble is None:
                X_trainEnsemble = np.zeros((X.shape[0], len(self.clfs)*classes))
                X_trainEnsemble[:, j * classes:(j + 1) * classes] = X_trainEnsemble_clf
            else:
                X_trainEnsemble[:, j * classes:(j + 1) * classes] = X_trainEnsemble_clf

        if self.original:
            X_trainEnsemble = np.append(X_trainEnsemble, X, axis=1)

        return X_trainEnsemble

    def generate_ensemble(self, X, y):
        print ('(%s) Blending Base Level Models with %s' %(time.asctime(time.localtime(time.time())),self.ensemble))

        self.ensemble.fit(X, y)

    def fit(self, X, y):
        if self.stratify:
            kf = StratifiedKFold(2)
        else:
            kf = KFold(2)

        self.kf = list(kf.split(X,y))
        self.fit_base(X[self.kf[0][0]], y[self.kf[0][0]])
        self.X_holdoutEnsemble = self.pred_base(X[self.kf[0][1]])
        self.generate_ensemble(self.X_holdoutEnsemble, y[self.kf[0][1]])

    def predict(self, X_test):
        self.X_testEnsemble = self.pred_base(X_test)

        return self.get_pred(self.ensemble, self.X_testEnsemble)






















