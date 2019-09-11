# -*- coding: utf-8 -*-
import os
import csv
import random


import numpy as np
import pandas as pd
import jieba


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


jieba.load_userdict("userdict.txt")
file_path = './review.csv'


def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    review_data = np.array(rows).tolist()
    random.shuffle(review_data)

    review_list = []
    sentiment_list = []
    for words in review_data:
        review_list.append(words[1])
        sentiment_list.append(words[0])

    return review_list, sentiment_list


def load_stopwords(file_path):
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
       stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words


def review_to_text(review):
    stop_words = load_stopwords('./stopwords.txt')
    # 去掉停用词
    review = jieba.cut(review)
    if stop_words:
        all_stop_words = set(stop_words)
        words = [w for w in review if w not in all_stop_words]

    return words


review_list, sentiment_list = load_corpus(file_path)
n = len(review_list) // 5

train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]
test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]

print('训练集数量： {}'.format(str(len(train_review_list))))
print('测试集数量： {}'.format(str(len(test_review_list))))

X_train = [' '.join(review_to_text(review)) for review in train_review_list]
y_train = train_sentiment_list

X_test = [' '.join(review_to_text(review)) for review in test_review_list]
y_test = test_sentiment_list

count_vec = CountVectorizer()
X_count = count_vec.fit_transform(X_train)

tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(X_train)


def MNB_count_Classifier():
    return Pipeline([
        ('count_vec', CountVectorizer()),
        ('mnb', MultinomialNB())
    ])


mnbc_clf = MNB_count_Classifier()
mnbc_clf.fit(X_train, y_train)

print('测试集准确率： {}'.format(mnbc_clf.score(X_test, y_test)))
a = mnbc_clf.predict(X_test).tolist()
err_list = []
for i in range(len(X_test)):
    data = {'sentiment': '', 'review': ''}
    if a[i] != y_test[i]:
        data['sentiment'] = y_test[i]
        data['review'] = X_test[i]

        err_list.append(data)

# for err in err_list:
#     print('测试集错误： {}'.format(err))
