from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np


def read_file():
    dataframe = pd.read_csv("a2a_train_round1.tsv", sep="\t", names=["Label", "Comment"])
    xtrain = dataframe.drop(columns=['Label'])
    ytrain = dataframe.drop(columns=['Comment'])
    return xtrain, ytrain


def remove_stopwords(df, column):
    stop = stopwords.words('english')
    df[column] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


def remove_punctuation(df, column):
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    df[column] = '|'.join(df[column].tolist()).translate(transtab).split('|')
    return df


def lower_case(df, column):
    df[column] = df[column].str.lower()
    return df


def lemmatize(df, column):
    """
    Not the best way for lemmatizing, I reckon
    """
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, 'n') for word in x.split()]))
    df[column] = df[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, 'v') for word in x.split()]))
    df[column] = df[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, 'a') for word in x.split()]))
    df[column] = df[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, 'r') for word in x.split()]))
    return df


def tfid_vectorize(x, column):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english', strip_accents='unicode')
    xvect = tf.fit_transform(x[column])
    return xvect


def classify(x, y):
    svm_cl = svm.SVC(kernel='linear')
    print("SVM: ", np.mean(cross_val_score(svm_cl, x, y, cv=5)))

    # dtc = DecisionTreeClassifier(random_state=0)
    # print("Decision tree classifier: ", cross_val_score(dtc, x, y, cv=10))
    #
    # rfc = RandomForestClassifier(n_estimators=500, criterion='entropy')
    # print("random forest classifier: ", cross_val_score(rfc, x, y, cv=10))

    # gbc = GradientBoostingClassifier(loss='deviance')
    # print("Gradient boosting classifier: ", cross_val_score(gbc, x, y, cv=3))

    lrc = LogisticRegression()
    print("Logistic regression classifier: ", np.mean(cross_val_score(lrc, x, y, cv=5)))

    lsc = LinearSVC()
    print("LinearSVC classifier: ", np.mean(cross_val_score(lsc, x, y, cv=5)))

    mnb = MultinomialNB()
    print("Multinomial NB classifier: ", np.mean(cross_val_score(lsc, x, y, cv=5)))


xtr, ytr = read_file()
xtr = remove_stopwords(xtr, "Comment")
xtr = remove_punctuation(xtr, "Comment")
xtr = lower_case(xtr, "Comment")
xtr = lemmatize(xtr, "Comment")
xv = tfid_vectorize(xtr, "Comment")
classify(xv, ytr)


