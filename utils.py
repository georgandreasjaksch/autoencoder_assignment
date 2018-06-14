# Utility methods to fit and evaluate a model and set-up respective pipelines
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
import joblib

# import the autoencoder and transformer classes
from autoencoder import Autoencoder
from densetransformer import DenseTransformer


def fit_and_evaluate_model(data_dir, filename, dimension_red="none"):
    '''
    Fits and evaluates a model. Dimension reduction type can be passed as a parameter
    :param data_dir: where the input data is stored (articles & labels)
    :param filename: filename (pickled data frame)
    :param dimension_red: type of dimensionality reduction {"none", "autoencoder", "svd", "pca"}
    '''
    df = pd.read_pickle(os.path.join(data_dir, filename))

    X = df['text']
    y = df['category']
    
    print('Loaded {0} rows with {1} distinct classes'.format(len(X), len(y.unique())))
    
    dimension_reduction_parameters = {"none", "autoencoder", "svd", "pca"}
    
    if dimension_red not in dimension_reduction_parameters:
        raise ValueError("dimension_reduction must be one of %r." % dimension_reduction_parameters)
     
    # set up a  pipeline for vectorizing, tf-idf transformation, applying dimensionality reduction & fitting
    text_clf = make_pipeline(dimension_red)
    
    # training/test split (75%:25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
     
    # model training and evaluation
    train_and_evaluate(text_clf, X_train, X_test, y_train, y_test)

    # print confusion matrix and report
    show_confusion_matrix(text_clf, X_test, y_test)

    
def make_pipeline(dimension_reduction="none"):
    '''
    Sets up a pipeline for the steps
    :param dimension_reduction: what type of dimensionality reduction shall be used {"none", "autoencoder", "svd", "pca"}
    '''
    # independent of dimensionality reduction vectorization and tf-idf is needed
    vectorizer = [('vect', CountVectorizer(max_features=2000, analyzer='word', stop_words='english')), ('tfidf', TfidfTransformer(use_idf=False))]
    # SVM classifier ('hinge')
    classifier =  [('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))]

    # Raw data classification (vectorize & classify)
    text_clf_raw = Pipeline(vectorizer + classifier)
                             
    # Vectorize, apply data compression with autoencoder and classify
    text_clf_autoenc = Pipeline(vectorizer + 
                             [('autoencoder', Autoencoder(n_features=2000,
                                 n_epochs=150,
                                 batch_size=8,
                                 enc_dimension=1000))] +
                                 classifier)

    # Vectorize, convert to dense, perform PCA and classify
    text_clf_pca = Pipeline(vectorizer + [('to_dense', DenseTransformer())] + 
                             [('pca', PCA(n_components=1000))] +
                             classifier)

    # Vectorize, perform SVD (as this supports sparse input)  and classify
    text_clf_svd = Pipeline(vectorizer + [('svd', TruncatedSVD(n_components=1000, n_iter=5, random_state=42))] +
                           classifier)

    if (dimension_reduction=="none"):
        return text_clf_raw
    elif (dimension_reduction=="svd"):
        return text_clf_svd
    elif (dimension_reduction=="pca"):
        return text_clf_pca
    elif (dimension_reduction=="autoencoder"):
        return text_clf_autoenc

# fit and evaluate/score the model
def train_and_evaluate(text_clf, X_train, X_test, y_train, y_test):
    '''
    :param text_clf: classifier from pipeline
    :param X_train: training features
    :param X_test: testing features
    :param y_train: training labels
    :param y_test: testing labels
    '''
    # fit the model...
    print('Training model')
    text_clf = text_clf.fit(X_train, y_train)
    print('Score on training data: {}'.format(text_clf.score(X_train, y_train)))
    # ... and evaluate
    print('Score on test data: {}'.format(text_clf.score(X_test, y_test)))


def show_confusion_matrix(text_clf, X_test, y_test):
    '''
    Prints classification report and confusion matrix
    :param text_clf: fitted classifier from pipeline
    :param X_test: testing features
    :param y_test: testing labels
    '''
    # get predictions from classifier
    predicted = text_clf.predict(X_test)
    # calculate "confusion"
    confusion = confusion_matrix(y_test, predicted)

    # classes are unique entries in y_test
    class_num = len(y_test.unique())
    
    df_confmat = pd.DataFrame(confusion,
                         index=[i for i in range(0, class_num)], columns=[i for i in range(0, class_num)])

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_confmat, annot=True)

    plt.title('Accuracy:{0:.3f}'.format(accuracy_score(y_test, predicted)))

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # print classification report
    print('Classification report:')
    print(classification_report(y_test, predicted,))

    plt.show()