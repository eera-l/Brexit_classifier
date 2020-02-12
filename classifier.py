import nltk
import pandas as pd


def read_file():
    dataframe = pd.read_csv("a2a_train_round1.tsv", sep="\t")
    Xtrain = dataframe.drop(index=0)
    Ytrain = dataframe[0]
    return Xtrain, Ytrain


Xtr, Ytr = read_file()