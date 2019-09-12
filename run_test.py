# -*- coding: utf-8 -*-
from native_bayes_sentiment_analyzer import SentimentAnalyzer


model_path = './data/bayes.pkl'
userdict_path = './data/userdict.txt'
stopword_path = './data/stopwords.txt'
corpus_path = './data/review.csv'


analyzer = SentimentAnalyzer(model_path=model_path, stopword_path=stopword_path, userdict_path=userdict_path)
text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
analyzer.analyze(text=text)
