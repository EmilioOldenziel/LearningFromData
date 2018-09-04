import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve,\
 confusion_matrix
from sklearn.model_selection import KFold

# hyperparams
PRINT_TEST_PROBABILITIES = True
# let's use the TF-IDF vectorizer
tfidf = True

# plot function that was copied function from sklearn 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Reads the trainset file using it's filename from the filesystem using utf-8 
# encoding.
# The document is read per line and stripped, the words and characters on the 
# list are placed in a list as tokens using the split() function.
# Tokens in index 3 and higher are stored as the document.
# If the use_sentiment variable is True, the sentiment in the form of postitive 
# and negative is used as labels.
# If the use_sentiment variable is not True (ELSE), the classes are used as 
# labels.
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return np.array(documents), np.array(labels)
    
# a dummy function that just returns its input
def identity(x):
    return x

# First, the trainset file is read and the X and Y vectors are set.
# Second, the X and Y vectors are split in a test and train set using 75% of 
# the data set as train set and 25% of the data set as test set.
X, Y = read_corpus('trainset.txt', use_sentiment=True)

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )

f = plt.figure(figsize=(8,8))
fold = 0
for train_index, test_index in KFold(n_splits=5).split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # The classifier consists of a Vectorizer that tranfers the words vector to a 
    # numeric vector that is used by the chosen classifier.
    # The resulting vectors are piped to the Multinomial Naive Bayes classifier, 
    # which receives the trainingset contraining the X and Y vectors.
    # The traingingset is used to fit/train a Naive Bayes model.
    classifier.fit(X_train, Y_train)

    # After the classifier's training process has completed, the resulting model is 
    # used to predict the testset's labels.
    Y_guess = classifier.predict(X_test)

    # To retrieve the classification accuracy, the predicted labels from testset 
    # are compared to the testset's ground truth labels.
    # The precentage of labels that are equal is the resulting accuracy.
    precisions = precision_score(Y_test, Y_guess, average=None)
    recalls = recall_score(Y_test, Y_guess, average=None)
    f1_scores = f1_score(Y_test, Y_guess, average=None)
    print("--- fold {fold} ---")
    for i, label in enumerate(np.unique(Y)):
        print(f"precision for {label}: {precisions[i]}")
        print(f"recall for {label}: {precisions[i]}")
        print(f"f1 score for {label}: {precisions[i]}")
    for averaging in ["micro", "macro"]:
        print(f"f1 scores: {f1_score(Y_test, Y_guess, average=averaging)}")

    if PRINT_TEST_PROBABILITIES:
        print(f"Log probability values for test set in fold {fold}")
        print(classifier.predict_log_proba(X_test))

    # display confusion matrix
    fold += 1
    f.add_subplot(320+fold)
    plot_confusion_matrix(confusion_matrix(Y_test, Y_guess), 
        classes=np.unique(Y), 
        normalize=True,
        title=f"Normalized confusion matrix \n for fold {fold}")
plt.show()