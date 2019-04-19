# top level imports
import os
import random
import numpy as np
import graphviz
from scipy.io import arff
import sklearn.datasets
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pydotplus
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.tree import export_graphviz
import sys
from numbers import Number
from collections import Set, Mapping, deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import timeit
import time
from sklearn.base import clone



zero_depth_bases = (basestring, Number, xrange, bytearray)
iteritems = 'iteritems'


# From StackOverflow post https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def fix_paths():
    if os.name == 'nt':
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        os.environ["PATH"] += os.pathsep + 'D:/omscs/ML/assignment_1/python-weka-wrapper-examples'
    # reset the PRNG
    random.seed(0)


def init_imports():
    import os
    import random
    import numpy as np
    import graphviz
    from scipy.io import arff
    import sklearn.datasets
    from itertools import product
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    import pydotplus
    from sklearn.externals.six import StringIO
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import GridSearchCV
    from sklearn.utils import shuffle
    from sklearn import preprocessing
    from sklearn.tree import export_graphviz
    import sys
    from numbers import Number
    from collections import Set, Mapping, deque
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from IPython.display import Image
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import StratifiedKFold
    import timeit
    import time
    from sklearn.base import clone


def load_digits(random_seed=None, path='data/digits.arff', scale=False, random_shuffle=True):
    data, meta = arff.loadarff(open(path, 'rb'))
    # set random state from value
    if random_seed is not None:
        np.random.seed(random_seed)
    # DIGITS leq4 or peq5
    val_a = (0, 1, 2, 3, 4)
    val_b = (5, 6, 7, 8, 9)
    yraw = [int(item) for item in data[meta.names()[-1]]]
    Xraw = [[item for item in row] for row, val in zip(data[meta.names()[:-1]], yraw) if val in val_a or val in val_b]
    ymod = [-1 if val in val_a else 1 for val in yraw if val in val_a or val in val_b]
    X = np.asarray(Xraw)
    y = np.asarray(ymod)
    # preprocess
    if scale:
        X = preprocessing.scale(X)
    # random shuffle
    if random_shuffle:
        X, y = shuffle(X, y)
    # group into dataset
    feature_names = [str(pair[0]) + "_" + str(pair[1]) for pair in product(range(8), range(8))]
    class_names = ['leq4', 'peq5']
    dataset = sklearn.datasets.base.Bunch(data=X, target=y, feature_names=feature_names, class_names=class_names)
    # Draw an element of data
    plt.imshow(X[0].reshape(8, 8), cmap='gray', interpolation='nearest', aspect='auto')
    print X[0]
    print str(val_a) if yraw[0] in val_a else str(val_b)
    plt.show()

    return dataset


def load_spambase(random_seed=None, path='data/spambase.arff', scale=False, random_shuffle=True):
    # import data from path
    data, meta = arff.loadarff(open(path, 'rb'))
    # set random state from value
    if random_seed is not None:
        np.random.seed(random_seed)
    # import spambase
    yraw = [int(item) for item in data[meta.names()[-1]]]
    Xraw = [[item for item in row] for row, val in zip(data[meta.names()[:-1]], yraw)]
    ymod = [-1 if val == 0 else 1 for val in yraw]
    # convert tu numpy arrays
    X = np.asarray(Xraw)
    y = np.asarray(ymod)
    # preprocess
    if scale:
        X = preprocessing.scale(X)
    # random shuffle
    if random_shuffle:
        X, y = shuffle(X, y)
    # group into dataset
    lines = [line.strip() for line in meta]
    feature_names = lines[0:-1]
    class_names = ['not_spam', 'spam']
    dataset = sklearn.datasets.base.Bunch(data=X, target=y, feature_names=feature_names, class_names=class_names)
    # Draw an element of data
    plt.imshow(X[0].reshape(1, -1), cmap='rainbow', interpolation='nearest', aspect='auto')
    print X[0]
    print 'not spam' if ymod[0] == -1 else 'spam'
    plt.show()

    return dataset

    
def test_classifier(classifier, dataset):
    X = dataset.data
    y = dataset.target
    scores = cross_val_score(classifier, X, y, cv=5)

    print "CV average = " + str(scores.mean())
    print "CV std = " + str(scores.std())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # test default classifier
    classifier.fit(X_train, y_train)

    print classifier


def export_tree(clf, dataset, path='dectree.pdf'):
    dot_data_string = StringIO()
    export_graphviz(clf,
                    out_file=dot_data_string,
                    feature_names=dataset.feature_names,
                    class_names=dataset.class_names,
                    filled=True,
                    rounded=True,
                    special_characters=True
                    )

    graph = pydotplus.graph_from_dot_data(dot_data_string.getvalue())
    graph.write_pdf(path)
    return graph.create_png()


def show_feature_importance(clf, dataset, shape=None):
    ndimport = clf.feature_importances_
    ndimport = ndimport.reshape(1, -1)

    if shape is not None:
        ndimport = ndimport.reshape(shape)

    plt.imshow(ndimport, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()

    df_importance = pd.DataFrame()
    df_importance["attribute"] = dataset.feature_names
    df_importance["importance"] = clf.feature_importances_
    return df_importance.sort_values(by="importance", ascending=0)[0:10]


def show_estimators_weights(clf, dataset):
    ndimport = clf.estimator_weights_
    ndimport = ndimport.reshape(1, -1)

    plt.imshow(ndimport, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()

    df_importance = pd.DataFrame()
    df_importance["weight"] = clf.estimator_weights_
    return df_importance[0:10]


def plot_learning_curve_draw(train_sizes, train_scores, test_scores, xlabel="", ylabel="", labels=["", ""]):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label=labels[0])
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label=labels[1])

    plt.legend(loc="best")
    return plt


def plot_learning_curve(classifier, dataset):
    X = dataset.data
    y = dataset.target
    train_sizes, train_scores, test_scores = learning_curve(
        classifier,
        X,
        y,
        train_sizes=[0.1 * t for t in range(1, 11)],
        cv=5)
    plot_learning_curve_draw(
        train_sizes,
        train_scores,
        test_scores,
        xlabel="Training examples",
        ylabel="Score",
        labels=["Training score", "Cross-validations score"])


def time_learning_curve(classifier, dataset, folds, random_state=0):
    X = dataset.data
    y = dataset.target

    # Save parameters
    train_sizes = []
    train_scores = []
    test_scores = []
    train_times = []
    predict_times = []

    nsplits = folds
    ntrain = 0

    # Get the test sizes
    train_sizes = [500 * t for t in range(1, 100) if (500 * t) < (X.shape[0] * (1. - 1. / nsplits))]

    for i in range(folds):

        # do random cv folds
        X_train_cv, X_test, y_train_cv, y_test = train_test_split(X, y, test_size=1. / folds, random_state=random_state)
        random_state += 1

        # print "fold=" + str(i)

        idx = 0
        for train_size in train_sizes:

            if train_size > X_train_cv.shape[0]:
                break
            if train_size not in train_sizes:
                train_sizes.append(train_size)

            idx += 1
            ntrain = max(ntrain, idx)
            # print "tsize=" + str(train_size)

            X_train_cv, y_train_cv = shuffle(X_train_cv, y_train_cv)

            X_train = X_train_cv[:train_size]
            y_train = y_train_cv[:train_size]

            clf = clone(classifier)

            start = time.time()
            clf.fit(X_train, y_train)
            end = time.time()
            train_times.append((end - start) / train_size)

            y_train_pred = clf.predict(X_train)
            score_train = accuracy_score(y_train_pred, y_train)

            start = time.time()
            y_test_pred = clf.predict(X_test)
            score_test = accuracy_score(y_test_pred, y_test)
            end = time.time()
            predict_times.append((end - start) / X_test.shape[0])

            train_scores.append(score_train)
            test_scores.append(score_test)

    train_sizes = np.array(train_sizes)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    train_times = np.array(train_times)
    predict_times = np.array(predict_times)

    train_scores = train_scores.reshape(ntrain, nsplits)
    test_scores = test_scores.reshape(ntrain, nsplits)
    train_times = 1000 * train_times.reshape(ntrain, nsplits)
    predict_times = 1000 * predict_times.reshape(ntrain, nsplits)

    plot_learning_curve_draw(
        train_sizes,
        train_times,
        predict_times,
        xlabel="Training examples",
        ylabel="Time per sample [ms]",
        labels=["Training time", "Predict time"])


def plot_complexity_curve(classifiers, dataset, depths, metric="Depth = "):
    X = dataset.data
    y = dataset.target

    results_size = []
    results_train = []
    results_test = []

    for classifier in classifiers:
        train_sizes, train_scores, test_scores = learning_curve(
            classifier,
            X, y,
            train_sizes=[0.1 * t for t in range(1, 11)],
            cv=5)
        results_size.append(train_sizes)
        results_train.append(train_scores)
        results_test.append(test_scores)

    fig, axes = plt.subplots(nrows=1, ncols=len(depths), figsize=(30, 4), sharey=True)
    labels = ["Training score", "Cross-validations score"]
    xlabel = "Training examples"
    ylabel = "Score"

    for train_sizes, train_scores, test_scores, ax, depth in zip(results_size, results_train, results_test, axes.flat,
                                                                 depths):
        ax.set_title(metric + str(depth))
        ax.set(xlabel=xlabel)
        ax.set(ylabel=ylabel)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid()

        ax.fill_between(train_sizes,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1,
                        color="g")

        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label=labels[0])
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label=labels[1])

        ax.legend(loc="best")


def plot_validation_curve(classifier, dataset, param_name, param_range, metric="Depth"):
    X = dataset.data
    y = dataset.target

    train_scores, test_scores = validation_curve(classifier,
                                                 X, y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(metric)
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)

    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy")
    plt.legend(loc="best")
    plt.show()


def find_best_pars(classifier, dataset, param_grid, noise_sigma=0):
    X = dataset.data
    y = dataset.target

    clf = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
    Xnoise = X + np.random.normal(0, noise_sigma, X.shape)
    clf.fit(Xnoise, y)
    print clf.best_params_
    return clf


def plot_noise_complexity_curves(classifiers, dataset, sigmas, depths, metric="Depth"):
    # classifiers is a list by sigma, then by complexity
    X = dataset.data
    y = dataset.target

    score_list = []
    rand_seed = -1

    for sigma, i_sigma in zip(sigmas, xrange(len(sigmas))):
        i_scores = []
        for depth, i_depth in zip(depths, xrange(len(depths))):
            rand_seed += 1
            X_train_iter = X + np.random.normal(0, sigma, X.shape)
            clf = classifiers[i_sigma][i_depth]
            cvscores = cross_val_score(clf, X_train_iter, y, cv=5)
            i_scores.append(cvscores)
        score_list.append(i_scores)

    fig, axes = plt.subplots(nrows=1, ncols=len(sigmas), figsize=(30, 4), sharey=True)
    labels = ["Cross-validations score"]
    xlabel = metric
    ylabel = "Score"

    for ax, sigma, iscores in zip(axes.flat, sigmas, score_list):
        ax.set_title("Sigma = " + str(sigma))
        ax.set(xlabel=xlabel)
        ax.set(ylabel=ylabel)

        test_scores = np.stack(iscores)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid()

        ax.fill_between(depths,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1,
                        color="g")

        ax.plot(depths, test_scores_mean, 'o-', color="g", label=labels[0])
        ax.legend(loc="best")


def view_hidden_layers(clf, idx=0, shapeplot=(8, 8), shape=(8, 8)):
    # Visualize inner layers
    fig, axes = plt.subplots(shapeplot[0], shapeplot[1], figsize=(8, 8))
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = clf.coefs_[idx].min(), clf.coefs_[idx].max()
    for coef, ax in zip(clf.coefs_[idx].T, axes.ravel()):
        ax.matshow(coef.reshape(shape[0], shape[1]), cmap=plt.cm.rainbow, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())