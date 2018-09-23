import numpy, json, argparse, itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from collections import Counter
numpy.random.seed(1337)


# plot function that was copied from sklearn 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
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


# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
    print('Reading in data from {0}...'.format(corpus_file))
    words = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            if binary_classes:
                if parts[1] in ['GPE', 'LOC']:
                    labels.append('LOCATION')
                else:
                    labels.append('NON-LOCATION')
            else:
                labels.append(parts[1])    
    print('Done!')
    return words, labels


# Read in word embeddings 
def read_embeddings(embeddings_file):
    print('Reading in embeddings from {0}...'.format(embeddings_file))
    embeddings = json.load(open(embeddings_file, 'r'))
    embeddings = {word: numpy.array(embeddings[word]) for word in embeddings}
    print('Done!')
    return embeddings


# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
    vectorized_words = []
    for word in words:
        try:
            vectorized_words.append(embeddings[word.lower()])
        except KeyError:
            vectorized_words.append(embeddings['UNK'])
    return numpy.array(vectorized_words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KerasNN parameters')
    parser.add_argument('data', metavar='named_entity_data.txt', type=str, help='File containing named entity data.')
    parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
    parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
    args = parser.parse_args()

    # Read in the data and embeddings
    X, Y = read_corpus(args.data, binary_classes=args.binary)
    print(Counter(Y))
    embeddings = read_embeddings(args.embeddings)

    # Transform words to embeddings
    X = vectorizer(X, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y)  # Use encoder.classes_ to find mapping of one-hot indices to string labels

    if args.binary:
        Y = numpy.where(Y == 1, [0, 1], [1, 0])

    f = plt.figure(figsize=(15,15))
    fold = 0
    scores = {'train': [], 'test': []}
    for train_index, test_index in list(KFold(n_splits=5,  shuffle=True).split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Define the properties of the perceptron model
        model = Sequential()
        model.add(Dense(input_dim=X.shape[1], units=Y.shape[1]))

        model.add(Activation("relu"))
        sgd = SGD(lr=0.01)
        loss_function = 'mean_squared_error'
        model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])

        # Train the perceptron
        model_result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),verbose=1, epochs=25, batch_size=32)

        Y_guess = model.predict(X_test)
        Y_guess = numpy.argmax(Y_guess, axis=1)
        Y_test = numpy.argmax(Y_test, axis=1)

        print(classification_report(Y_test, Y_guess))

        # display confusion matrix
        fold += 1
        f.add_subplot(320+fold)
        plot_confusion_matrix(confusion_matrix(Y_test, Y_guess), 
            classes=['GPE', 'ORG', 'PERSON', 'CARDINAL', 'DATE', 'LOC'], # classes=["NON-LOCATION", "LOCATION"],
            title=f"Normalized confusion matrix \n for fold {fold}")

        scores['train'].append(model_result.history['acc'][-1])
        scores['test'].append(model_result.history['val_acc'][-1])

    train_avg = numpy.mean(scores["train"])
    test_avg = numpy.mean(scores["test"])
    print(f"Train average: {train_avg}")
    print(f"Test average: {test_avg}")

    plt.show()