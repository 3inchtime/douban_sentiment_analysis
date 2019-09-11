# -*- coding: utf-8 -*-
import re
import pickle

import numpy as np
import jieba


class SentimentAnalyzer(object):
    def __init__(self, model_path, userdict_path, stopword_path):
        self.clf = None
        self.vectorizer = None
        self.tfidftransformer = None
        self.model_path = model_path
        self.stopword_path = stopword_path
        self.userdict_path = userdict_path
        self.stop_words = []
        self.tokenizer = jieba.Tokenizer()
        self.initialize()

    def initialize(self):
        with open(self.stopword_path, encoding='UTF-8') as words:
            self.stop_words = [i.strip() for i in words.readlines()]

        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
            self.clf = model['clf']
            self.vectorizer = model['vectorizer']
            self.tfidftransformer = model['tfidftransformer']
        if self.userdict_path:
            self.tokenizer.load_userdict(self.userdict_path)

    def replace_text(self, text):
        text = re.sub('((https?|ftp|file)://)?[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|].(com|cn)', '', text)
        text = text.replace('\u3000', '').replace('\xa0', '').replace('”', '').replace('"', '')
        text = text.replace(' ', '').replace('↵', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('）', '')
        return text

    def predict_score(self, lists):
        docs = [self.__cut_word(sentence) for sentence in lists]
        new_tfidf = self.tfidftransformer.transform(self.vectorizer.transform(docs))
        predicted = self.clf.predict_proba(new_tfidf)
        predicted_new_decimals = np.around(predicted, decimals=3)
        return predicted_new_decimals

    def __cut_word(self, sentence):
        words = [i for i in self.tokenizer.cut(sentence) if i not in self.stop_words]
        result = ' '.join(words)
        return result

    def analyze(self, text):
        text_corpus = self.replace_text(text)
        text_corpus = [text_corpus]
        predicted_new_decimals = self.predict_score(text_corpus)

        neg = predicted_new_decimals[0][0]
        pos = predicted_new_decimals[0][1]

        if neg > pos:
            print('"{}" 验证结果为： {}'.format(text, '差评'))
        else:
            print('"{}" 验证结果为： {}'.format(text, '好评'))
