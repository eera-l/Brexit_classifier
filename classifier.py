from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import numpy as np


def read_file():
    dataframe = pd.read_csv("a2a_train_round1.tsv", sep="\t", names=["Label", "Comment"])
    xtrain = dataframe.drop(columns=['Label'])
    ytrain = dataframe.drop(columns=['Comment'])
    return xtrain, ytrain


def remove_stopwords(df, column):
    stop = stopwords.words('english')
    stop.remove('against')
    stop.append('the')
    stop.append('a')
    df[column] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
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


def stem(df, column):
    stemmer = SnowballStemmer("english")
    df[column] = df[column].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    return df


def perform_lda(x):
    texts = x.values.tolist()
    words = []
    for text in texts:
        words.append([x for x in text[0].split()])
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(word) for word in words]
    # ldamodel = LdaModel(corpus, num_topics=2, id2word=dictionary,
    #                        passes=50)
    # ldamodel.save('lda.model')
    ldamodel = LdaModel.load('lda.model')
    print(ldamodel.print_topics(num_topics=2, num_words=15))


def train_classifier(clf, x, y):
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer='word', ngram_range=(1,4), stop_words='english', strip_accents='unicode'),
        clf)
    pipeline.fit(x, y)
    return pipeline


def classify(classifiers, x, y):
    train_scores = []
    test_scores = []
    x = x['Comment']
    y = y['Label']
    for clf in classifiers:
        pipeline = train_classifier(clf, x, y)

        scores = cross_validate(pipeline, x, y, cv=5,
                                scoring=('accuracy', 'f1'),
                                return_train_score=True)
        name = pipeline.steps[1][0]

        train_scores.append((np.mean(scores['train_accuracy']), np.mean(scores['train_f1']), name))
        test_scores.append((np.mean(scores['test_accuracy']), np.mean(scores['test_f1']), name))

    return train_scores, test_scores


def initialize_classifiers(x, y):

    dmc = DummyClassifier(strategy='stratified')
    svc = svm.SVC(kernel='linear')
    lr = LogisticRegression(solver='newton-cg', fit_intercept=True)
    lsvc = LinearSVC(max_iter=2000)
    mnb = MultinomialNB(alpha=2.0)

    classifiers = [dmc, svc, lr, lsvc, mnb]

    train_scores, test_scores = classify(classifiers, x, y)
    for train, test in zip(train_scores, test_scores):
        print("Train: ", train, " test: ", test)


xtr, ytr = read_file()
xtr = remove_punctuation(xtr, "Comment")
xtr = lower_case(xtr, "Comment")
xtr = remove_stopwords(xtr, "Comment")
xtr = lemmatize(xtr, "Comment")
#perform_lda(xtr)
# xtr = stem(xtr, "Comment")
initialize_classifiers(xtr, ytr)


