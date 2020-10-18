import progressbar
import pandas as pd
import cv2 as cv
import numpy as np
from scipy.stats import randint as sp_randint
from skimage import exposure
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import sklearn.svm as svm
import sklearn.neighbors as nb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# number of images in data
n = len(pd.read_csv('train.csv'))

# Base dimensions to resize images to
dim = (291, 238)

data = []

print('RESIZING AND FILTERING IMAGES:')
bar = progressbar.ProgressBar(maxval=n, \
    widgets=[progressbar.Bar('*', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(n):

    img = cv.imread('train/train/img-' + str(i) + '.jpeg')

    # apply histogram equalization
    res = exposure.equalize_hist(img)

    # apply local contrast enhancement
    res = exposure.equalize_adapthist(res)

    # resize
    res = cv.resize(res, dim, interpolation=cv.INTER_LANCZOS4)

    data.append(res)
    bar.update(i+1)
bar.finish()

# flatten array
X = []
for mat in data:
    X.append(mat.flatten())

# Training-test split
y = list(pd.read_csv('data.csv').drop('filename', axis=1)['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Rescale data
scaler = StandardScaler()

# Fit on training set
scaler.fit(X)

# Apply transform to both the training set and the test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Execute Principal Component Analysis
# Make an instance of the Model
pca = PCA(.95)
pca.fit(X_train)

# Project data
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Oversample minority class covid and undersample the other classes
count_min = 200
over = SMOTE(sampling_strategy={'covid':count_min})
under = RandomUnderSampler(sampling_strategy={'normal':200,
                                              'bacterial':200,
                                              'viral':200 })
X_train, y_train = over.fit_sample(X_train, y_train)

# Optimize KNN hyperparamters
search_space = {
        "n_neighbors": Integer(1, 10),
        "algorithm": Categorical(['ball_tree', 'kd_tree', 'brute']),
        "p": Integer(1, 10),
        "metric": Categorical(['minkowski', 'chebyshev', 'manhattan'] )
    }

knn_clf = nb.KNeighborsClassifier()
knn_bayes_search = BayesSearchCV(knn_clf, search_space, n_iter=35, scoring="balanced_accuracy", cv=5)
knn_bayes_search.fit(X_train, y_train)

# Optimize RF hyperparamters
search_space = {
        'bootstrap': Categorical([True, False]),
        'max_depth': Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]),
        'max_features': Categorical(['sqrt', 'log2']),
        'min_samples_leaf': Categorical([1, 2, 4]),
        'min_samples_split': Categorical([2, 5, 10])
}

rf_clf = RandomForestClassifier()
rf_bayes_search = BayesSearchCV(rf_clf, search_space, n_iter=50, scoring="balanced_accuracy", cv=5)
rf_bayes_search.fit(X_train, y_train)

# Optimize SVM hyperparamters
search_space = {
        'estimator__C': Categorical([.1, .2, .5, 1, 2, 5, 10, 20, 50, 70, 100]),
        'estimator__gamma': Categorical([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100,
                                         5e-9, 5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5, 50])
}

svm_clf = OneVsRestClassifier(svm.SVC())
svm_bayes_search = BayesSearchCV(svm_clf, search_space, n_iter=35, scoring="balanced_accuracy", cv=5)
svm_bayes_search.fit(X_train, y_train)

# Optimize LR hyperparamters
search_space = {
        'penalty': ['l2', 'none'],
        'C': Categorical(np.logspace(-4, 4, 50)),
        'solver': Categorical(['saga'])
    }

lr_clf = LogisticRegression()
lr_bayes_search = BayesSearchCV(lr_clf, search_space, n_iter=35, scoring="balanced_accuracy", cv=5)
lr_bayes_search.fit(X_train, y_train)

# Optimize MLP for hyperparamters
parameter_space = {
        'hidden_layer_sizes': [(sp_randint.rvs(100,600,1), sp_randint.rvs(100,600,1),),
                               (sp_randint.rvs(100,600,1),)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [1e-5, 1e-4, .05],
        'learning_rate': ['constant','adaptive', 'invscaling']
    }

mlp_clf = MLPClassifier()
mlp_bayes_search = RandomizedSearchCV(mlp_clf, parameter_space, n_iter=50, scoring="balanced_accuracy", cv=5)
mlp_bayes_search.fit(X_train, y_train)
