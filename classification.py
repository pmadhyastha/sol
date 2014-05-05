#!/usr/bin/python

from __future__ import division

import sys
import glob
import numpy as np
import sklearn as sk
import sklearn.datasets as skd
import sklearn.feature_extraction.text as skfet
import sklearn.linear_model as sklm
import sklearn.svm as sksvm

clfier = sys.argv[1]

train_dat = skd.fetch_20newsgroups(subset='train', categories=None)

y, tr_labels = train_dat.target, train_dat.target_names

#tfidf_feats = skfet.CountVectorizer(sublinear_tf=True, max_df=0.5)

count_feats = skfet.CountVectorizer()
#X = tfidf_feats.fit_transform(train_dat.data)
X = count_feats.fit_transform(train_dat.data)

def loadsamples():
    data = []
    serial = []
    for fl in glob.glob('testfls/*'):
        serial.append(int(fl.split('/')[1]))
        data.append(''.join(l for l in [s for s in open(fl)]))

#    data_feat = tfidf_feats.fit_transform(data)
    data_feat = count_feats.fit_transform(np.asarray(data))
    return (data_feat, serial)


def prediction(clf):
    (sample_dat, sample_serial) = loadsamples()
    print sample_dat[0]
    pred = clf.predict(sample_dat)
    count = 0
    for p in pred:
        print sample_serial[count], tr_labels[p]

def classify(clf):
    clf.fit(X, y)
    return clf


if clfier == 'ridge':
    clf = sklm.RidgeClassifier(tol=1e-2, solver="lsqr")
elif clfier == 'perceptron':
    clf = sklm.Perceptron(n_iter=100)
elif clfier == 'svm':
    clf = sksvm.LinearSVC(loss='l2', penalty='l2', tol=1e-3)
elif clfier == 'graddesc':
    clf = sklm.SGDClassifier(alpha=0.0001, n_iter=100, penalty='l2')

clsff = classify(clf)
prediction(clf)

