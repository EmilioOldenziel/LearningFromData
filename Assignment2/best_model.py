import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# know training set
TRAINING_SET_FILENAME = "trainset.txt"

# blink test set
TEST_SET_FILENAME = "trainset.txt"

def read_corpus(corpus_file, use_sentiment):
    st = PorterStemmer()
    stop = stopwords.words('english')
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            
            # remove stopwords    
            doc = [token for token in tokens[3:] if token not in stop]
        
            # porter stemmer
            doc = [st.stem(word) for word in doc]

            documents.append(doc)

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

# read known training set
X_train, Y_train = read_corpus(, use_sentiment=False)

# read blind test set
X_test, Y_test = read_corpus('trainset.txt', use_sentiment=False)

# use TD-IDF vectors
vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)

# Search range for alpha
params = {"cls__alpha": np.arange(0.5, 0.6, 0.01)}

# use Naive bayes as classifier
clf = Pipeline( [('vec', vec), ('cls', MultinomialNB())])

# find optimal alpha
alpha = 0
for measure in ["f1_micro", "f1_macro"]:
    print(f"+++++++++++ Finding alpha for optimal {measure} score +++++++++++")
    GS = GridSearchCV(clf, params, cv=5, scoring=measure, 
        n_jobs=4, verbose=1, return_train_score=True)
    GS.fit(X_train, Y_train)

    df = pd.DataFrame(GS.cv_results_)
    best_setting = df.sort_values(by="mean_test_score", ascending=False).iloc[0]
    print("Best fit found for:")
    print(best_setting)
    alpha += best_setting["params"]["cls__alpha"]

# lets meet in the middle between micro and macro
alpha /= 2

# train on full trainingset
print(f"use alpha {alpha:.2f} for training the model on all data")
clf = Pipeline( [('vec', vec), ('cls', MultinomialNB(alpha=alpha))])
clf.fit(X_train,Y_train)

# predict test set and print scores
print("results of the provided test set")
print(classification_report(clf.predict(X_test), Y_test))