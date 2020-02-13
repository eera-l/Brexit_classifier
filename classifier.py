from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from scipy import sparse
from sklearn.svm import LinearSVC
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import numpy as np


def read_labels(y):
    results = []
    for row in y:
        labels = row.split("/")
        labels = [num for num in labels if num is not "-1"]
        labels = list(map(int, labels))
        labels[0] = labels[0] * 1.2 # Assign larger weight to original annotator
        mean = np.mean(labels)
        results.append(1 if mean >= 0.5 else 0)
    ydf = pd.DataFrame(results, columns='Label')
    return ydf



def read_file():
    dataframe = pd.read_csv("a2a_train_final.tsv", sep="\t", names=["Label", "Comment"])
    xtrain = dataframe.drop(columns=['Label'])
    ytrain = dataframe.drop(columns=['Comment'])
    ytrain = ytrain['Label']
    ytrain = read_labels(ytrain)
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


def add_length(x):
    scaler = MinMaxScaler()
    data = np.array([len(line) for line in x]).reshape(-1, 1)
    data = scaler.fit_transform(data)
    return data


def tfidf_vectorize(x):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), stop_words='english', strip_accents='unicode')
    x_vect = tfidf.fit_transform(x)
    return x_vect


def combine_tfidf_and_length(x_t, x_l):
    x_new = sparse.hstack((x_t, x_l))
    return x_new


def train_classifier(clf, x, y):
    clf.fit(x, y)
    name = clf.__class__.__name__
    joblib.dump(clf, name + '.pkl')
    return clf


def classify(classifiers, x, y):
    train_scores = []
    test_scores = []
    y = y['NewLabel']
    for clf in classifiers:
        clf = train_classifier(clf, x, y)

        scores = cross_validate(clf, x, y, cv=5,
                                scoring=('accuracy', 'f1'),
                                return_train_score=True)
        name = clf.__class__.__name__
        train_scores.append((name, np.mean(scores['train_accuracy']), np.mean(scores['train_f1'])))
        test_scores.append((name, np.mean(scores['test_accuracy']), np.mean(scores['test_f1'])))

    return train_scores, test_scores


def initialize_classifiers(x, y):

    dmc = DummyClassifier(strategy='stratified')
    lsvc = LinearSVC()
    mnb = MultinomialNB(alpha=2.0)
    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()

    classifiers = [dmc, lsvc, mnb, rfc, gbc]

    y = y['Label']

    for clf in classifiers:
        compare_with_gscv(clf, x, y)

    # train_scores, test_scores = classify(classifiers, x, y)
    # for train, test in zip(train_scores, test_scores):
    #     print("Train: ", train, " test: ", test)


def compare_with_gscv(clf, x, y):
    name = clf.__class__.__name__
    param_grid = {}
    if name is 'DummyClassifier':
        return
    elif name is 'LinearSVC':
        param_grid = {
            'penalty': ['l1', 'l2'],
            'max_iter': [500, 1000, 1500],
            'loss': ['hinge', 'squared_hinge']
        }
    elif name is 'MultinomialNB':
        param_grid = {
            'alpha': [0.5, 1.0, 1.5],
            'fit_prior': [True, False],
        }
    elif name is 'RandomForestClassifier':
        param_grid = {
            'min_samples_split': [3, 5, 10],
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 15, 25],
            'max_features': [3, 5, 10, 20]
        }
    elif name is 'GradientBoostingClassifier':
        param_grid = {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.05],
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 15, 25]
        }

    refit_score = 'accuracy_score'
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                               cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(x, y)
    print('\nScore {} classifier optimized for {} on the test data:'.format(name, refit_score))
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.alpha)
    return grid_search


xtr, ytr = read_file()
xtr = remove_punctuation(xtr, "Comment")
xtr = lower_case(xtr, "Comment")
xtr = remove_stopwords(xtr, "Comment")
xtr = lemmatize(xtr, "Comment")
xtr = stem(xtr, "Comment")
x_length = add_length(xtr['Comment'])
x_tfidf = tfidf_vectorize(xtr['Comment'])
x_new = combine_tfidf_and_length(x_tfidf, x_length)
#perform_lda(xtr)
initialize_classifiers(x_new, ytr)


