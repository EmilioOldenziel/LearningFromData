import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
    
if __name__ == '__main__':
    print("Reading data")
    # read known training set
    X, Y = read_corpus("trainset.txt", use_sentiment=True)

    lb = LabelBinarizer()
    Y = lb.fit_transform(Y)

    # use TD-IDF vectors
    vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)

    # Search range for alpha
    params = {
        "cls__n_clusters": np.arange(2,20),
        "cls__init": ["k-means++", "random"],
        "cls__n_init": [20],
        "cls__max_iter": [pow(b,p) for p, b in enumerate([10]*5)], 
        "cls__n_jobs": [4]
        }

    # use Naive bayes as classifier
    clf = Pipeline( [('vec', vec), ('cls', KMeans())])

    # find optimal alpha
    measure = "f1_micro"
    GS = GridSearchCV(clf, params, cv=5, scoring=measure, 
        n_jobs=1, verbose=1, return_train_score=True)
    GS.fit(X, Y)

    df = pd.DataFrame(GS.cv_results_)
    df.to_csv("best_results_kmeans.csv")

