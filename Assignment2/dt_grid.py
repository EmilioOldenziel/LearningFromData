import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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

X, Y = read_corpus('trainset.txt', use_sentiment=False)

vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)

params = {
    "cls__max_depth": range(20,21), 
    # "cls__min_samples_split": range(2,10),
    # "cls__min_samples_leaf": range(1,20),
    # "cls__max_leaf_nodes": range(6,50),
}

clf = Pipeline( [('vec', vec), ('cls', DecisionTreeClassifier())] )

GS = GridSearchCV(clf, params, cv=5,
                       scoring="f1_micro", n_jobs=64)

GS.fit(X,Y)

df = pd.DataFrame(GS.cv_results_)
df.to_csv("DT_results")