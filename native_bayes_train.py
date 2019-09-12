# -*- coding: utf-8 -*-
import os
import csv
import random
import pickle

import numpy as np
import jieba


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


jieba.load_userdict("./data/userdict.txt")


file_path = './data/review.csv'
model_export_path = './data/bayes.pkl'
stopword_path = './data/stopwords.txt'


def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    review_data = np.array(rows).tolist()
    # 打乱数据顺序
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


# jieba分词
def review_to_text(review):
    stop_words = load_stopwords(stopword_path)
    review = jieba.cut(review)
    all_stop_words = set(stop_words)
    # 去掉停用词
    review_words = [w for w in review if w not in all_stop_words]

    return review_words


review_list, sentiment_list = load_corpus(file_path)
n = len(review_list) // 5

train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]
test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]

print('训练集数量： {}'.format(str(len(train_review_list))))
print('测试集数量： {}'.format(str(len(test_review_list))))

review_train = [' '.join(review_to_text(review)) for review in train_review_list]
sentiment_train = train_sentiment_list

review_test = [' '.join(review_to_text(review)) for review in test_review_list]
sentiment_test = test_sentiment_list


vectorizer = CountVectorizer(max_df=0.8, min_df=3)

tfidftransformer = TfidfTransformer()

# 先转换成词频矩阵，再计算TFIDF值
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(review_train))
# 朴素贝叶斯中的多项式分类器
clf = MultinomialNB().fit(tfidf, sentiment_train)

# 将模型保存pickle文件
with open(model_export_path, 'wb') as file:
    d = {
        "clf": clf,
        "vectorizer": vectorizer,
        "tfidftransformer": tfidftransformer,
    }
    pickle.dump(d, file)

print("训练完成")
