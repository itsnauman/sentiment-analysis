import numpy as np
import random

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

from sklearn import svm
from sklearn.model_selection import GridSearchCV

import random

with open('./data/vocab.txt', 'r') as fp:
    vocab_list = fp.read().split('\n')
    vocab = {word: count for count, word in enumerate(vocab_list)}


def ngrams(w_input, n):
    """
    Generate a set of n-grams for more complex features
    """
    output = []

    for i in range(len(w_input) - n + 1):
        output.append(w_input[i: i + n])
    return [''.join(x) for x in output]


def tokenize_sentence(sentence):
    """
    Group tokenized sentences and it's corresponding sentiment into a tuple
    """
    stop = stopwords.words('english')
    lmtzr = WordNetLemmatizer()

    token_words = word_tokenize(sentence)
    sentiment = token_words[-1]
    tokens = [lmtzr.lemmatize(w.lower()) for w in token_words if w not in stop and w.isalpha() and len(w) >= 3]
    bi_grams = ngrams(tokens, 2)

    return (tokens + bi_grams, sentiment)


def extract_features(sentence):
    """
    Extract features from tokenized sentence and build a feature vector
    """
    feature_vector = [0] * len(vocab)
    tokens, sentiment = tokenize_sentence(sentence)
    for word in tokens:
        if word in vocab:
            feature_vector[vocab[word]] = 1
    return (feature_vector, sentiment)


def get_sentences_from_files(files):
    """
    Load corpus from file
    """
    sentences = []
    for file_name in files:
        with open(file_name, 'r') as fp:
            sentences.extend(fp.readlines())
    return sentences


def build_data_set(sentences):
    """
    Build feature vector X and output vector y for training the classifier
    """
    X = []
    y = []

    for each_sentence in sentences:
        features, sentiment = extract_features(each_sentence)
        X.append(features)
        y.append(sentiment)

    X = np.array(X)
    y = np.array(y)

    return X, y


def optimize_params(X, y, clf, param_grid):
    """
    Find optiumum values for C, gamma and degree using GridSearch
    and Cross Fold Validation
    """
    clf = GridSearchCV(clf, param_grid, cv=2, n_jobs=2)
    clf.fit(X, y)
    return clf.best_params_

if __name__ == '__main__':
    sentences = get_sentences_from_files(['./data/sentences_labelled.txt'])
    X, y = build_data_set(sentences)

    # Training dataset
    train_X = X[:2000]
    train_y = y[:2000]

    # Cross-validation dataset
    cross_X = X[2000: 2400]
    cross_y = y[2000: 2400]

    # Testing dataset
    test_X = X[2400:]
    test_y = y[2400:]


    svc = svm.SVC(kernel='poly', C=10, gamma=0.1, degree=2)
    svc.fit(train_X, train_y)
    print(svc.score(cross_X, cross_y))
