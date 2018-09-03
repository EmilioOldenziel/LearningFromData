from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

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

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# First, the trainset file is read and the X and Y vectors are set.
# Second, the X and Y vectors are split in a test and train set using 75% of 
# the data set as train set and 25% of the data set as test set.
X, Y = read_corpus('trainset.txt', use_sentiment=True)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

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


# The classifier consists of a Vectorizer that tranfers the words vector to a 
# numeric vector that is used by the chosen classifier.
# The resulting vectors are piped to the Multinomial Naive Bayes classifier, 
# which receives the trainingset contraining the X and Y vectors.
# The traingingset is used to fit/train a Naive Bayes model.
classifier.fit(Xtrain, Ytrain)

# After the classifier's training process has completed, the resulting model is 
# used to predict the testset's labels.
Yguess = classifier.predict(Xtest)

# To retrieve the classification accuracy, the predicted labels from testset 
# are compared to the testset's ground truth labels.
# The precentage of labels that are equal is the resulting accuracy.
print(accuracy_score(Ytest, Yguess))

