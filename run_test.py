# -*- coding: utf-8 -*-
import random
import numpy as np
import csv
from native_byes_sentiment_analyzer import SentimentAnalyzer


model_path = './data/byes.pkl'
userdict_path = './data/userdict.txt'
stopword_path = './data/stopwords.txt'
corpus_path = './data/review.csv'


def load_corpus(corpus_path):
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    review_data = np.array(rows).tolist()
    random.shuffle(review_data)

    review_list = []
    for words in review_data[:20]:
        review_list.append(words[1])

    return review_list


analyzer = SentimentAnalyzer(model_path=model_path, stopword_path=stopword_path, userdict_path=userdict_path)

review_list = load_corpus(corpus_path=corpus_path)
for review in review_list:
    analyzer.analyze(text=review)
