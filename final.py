import pandas as pd
from prep_vars import STOP_WORDS, PUNCTUATION, SUFFIX
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import nltk
import numpy as np
import matplotlib.pyplot as plt


dataset_train = pd.read_csv("./ilur-news-classification/train.csv")
dataset_test = pd.read_csv("./ilur-news-classification/test.csv")

X_train = dataset_train["Text"]
Y_train = dataset_train["Category"]

X_test = dataset_test["Text"]


def preprocessing(text):
    # Lowercase
    text = text.lower()

    # Punctuation
    for ch in PUNCTUATION:
        text = text.replace(ch, "")
    l = text.split()
    without_stopwords = [w for w in l if w not in STOP_WORDS]
    without_stopwords = " ".join(without_stopwords)

    # Stemming
    # tokens = []
    # for word in without_stopwords:
    #    if len(word) >= 2 and word[-2] == "ո" and word[-1] == "ւ":
    #       word = word.replace(word[-2:], "")
    #    if word[-1] == "ը" or word[-1] == "ի":
    #        word = word.replace(word[-1], "")
    #  # else:
    #  #     for it in SUFFIX:
    #  #         if len(it) == 2 and it in word[-2:]:
    #  #             word = word.replace(it, "")
    #  #         elif len(it) == 3 and it in word[-3:]:
    #  #             word = word.replace(it, "")
    #  #         elif len(it) == 4 and it in word[-4:]:
    #  #             word = word.replace(it, "")
    #  #         elif len(it) == 6 and it in word[-6:]:
    #  #             word = word.replace(it, "")
    #  #         if len(it) == 7 and it in word[-7:]:
    #  #             word = word.replace(it, "")
       # tokens.append(word)
    # tokens = " ".join(tokens)
    return without_stopwords


# Getting preprocessed train dataset
preprocessed = pd.Series([], dtype=pd.StringDtype())
# preprocessed_test = pd.Series([], dtype=pd.StringDtype())
for elem in X_train:
    elem = pd.Series(preprocessing(elem))
    preprocessed = preprocessed.append(elem, ignore_index=True)

# for elem in X_test:
#     elem = pd.Series(preprocessing(elem))
#     preprocessed_test = preprocessed_test.append(elem, ignore_index=True)
# preprocessed = preprocessing(X_train)

vectorizer = TfidfVectorizer()
max_abs_scaler = MaxAbsScaler()
x_full = vectorizer.fit_transform(preprocessed)
x_full = max_abs_scaler.fit_transform(x_full)
x_test = vectorizer.transform(X_test)
x_test = max_abs_scaler.transform(x_test)
x_train, x_valid, y_train, y_valid = train_test_split(preprocessed, Y_train, test_size=0.2, random_state=0)
#vector_x_train = vectorizer.fit_transform(x_train)
vector_x_valid = vectorizer.transform(x_valid)
clf = LogisticRegressionCV(cv=10, random_state=0, max_iter=200, n_jobs=4).fit(x_full, Y_train)
pred = clf.predict(x_test)
df = pd.DataFrame(pred, columns=["Category"])
df.to_csv("cross_val_result.csv", mode='a')
print(clf.score(vector_x_valid, y_valid))
