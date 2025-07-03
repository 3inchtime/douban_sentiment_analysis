# -*- coding: utf-8 -*-
"""
豆瓣影评情感分析器

基于朴素贝叶斯算法实现的中文文本情感分析工具，主要用于分析豆瓣影评的情感倾向。
支持自定义词典和停用词，使用TF-IDF特征提取和多项式朴素贝叶斯分类器。

Author: 3inchtime
Date: 2019
"""

import re
import pickle

import numpy as np
import jieba


class SentimentAnalyzer(object):
    """
    情感分析器类
    
    使用训练好的朴素贝叶斯模型对中文文本进行情感分析，
    能够判断文本的正面情感和负面情感概率。
    """
    
    def __init__(self, model_path, userdict_path, stopword_path):
        """
        初始化情感分析器
        
        Args:
            model_path (str): 训练好的模型文件路径（pickle格式）
            userdict_path (str): 用户自定义词典文件路径
            stopword_path (str): 停用词文件路径
        """
        # 模型相关组件
        self.clf = None  # 朴素贝叶斯分类器
        self.vectorizer = None  # 词频向量化器
        self.tfidftransformer = None  # TF-IDF转换器
        
        # 文件路径配置
        self.model_path = model_path
        self.stopword_path = stopword_path
        self.userdict_path = userdict_path
        
        # 停用词列表
        self.stop_words = []
        
        # jieba分词器
        self.tokenizer = jieba.Tokenizer()
        
        # 初始化模型和词典
        self.initialize()

    def initialize(self):
        """
        初始化模型和相关资源
        
        加载停用词列表、训练好的模型组件（分类器、向量化器、TF-IDF转换器）
        以及用户自定义词典。
        """
        # 加载停用词列表
        with open(self.stopword_path, encoding='UTF-8') as words:
            self.stop_words = [i.strip() for i in words.readlines()]

        # 加载训练好的模型
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
            self.clf = model['clf']  # 朴素贝叶斯分类器
            self.vectorizer = model['vectorizer']  # 词频向量化器
            self.tfidftransformer = model['tfidftransformer']  # TF-IDF转换器
            
        # 加载用户自定义词典（如果存在）
        if self.userdict_path:
            self.tokenizer.load_userdict(self.userdict_path)

    def replace_text(self, text):
        """
        文本预处理和清洗
        
        过滤掉文本中的URL链接、特殊字符、空白字符等无关内容，
        并按照中文标点符号将文本分割成句子列表。
        
        Args:
            text (str): 原始文本
            
        Returns:
            list: 分割后的句子列表
        """
        # 移除URL链接
        text = re.sub('((https?|ftp|file)://)?[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|].(com|cn)', '', text)
        
        # 移除各种特殊字符和空白字符
        text = text.replace('\u3000', '').replace('\xa0', '').replace('"', '').replace('"', '')
        text = text.replace(' ', '').replace('↵', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('）', '')
        
        # 按照中文标点符号分割文本为句子
        text_corpus = re.split('[！。？；……;]', text)
        return text_corpus

    def predict_score(self, text_corpus):
        """
        情感分析预测
        
        对预处理后的文本进行情感分析，返回正面和负面情感的概率。
        
        Args:
            text_corpus (list): 预处理后的句子列表
            
        Returns:
            numpy.ndarray: 情感预测概率矩阵，每行包含[负面概率, 正面概率]
        """
        # 对每个句子进行分词处理
        docs = [self.__cut_word(sentence) for sentence in text_corpus]
        
        # 将分词结果转换为TF-IDF特征向量
        new_tfidf = self.tfidftransformer.transform(self.vectorizer.transform(docs))
        
        # 使用朴素贝叶斯分类器预测情感概率
        predicted = self.clf.predict_proba(new_tfidf)
        
        # 四舍五入，保留三位小数
        result = np.around(predicted, decimals=3)
        return result

    def __cut_word(self, sentence):
        """
        中文分词处理（私有方法）
        
        使用jieba分词器对句子进行分词，并过滤掉停用词。
        
        Args:
            sentence (str): 待分词的句子
            
        Returns:
            str: 分词后用空格连接的字符串
        """
        # 使用jieba分词，并过滤停用词
        words = [i for i in self.tokenizer.cut(sentence) if i not in self.stop_words]
        
        # 将分词结果用空格连接
        result = ' '.join(words)
        return result

    def analyze(self, text):
        """
        文本情感分析主方法
        
        对输入文本进行完整的情感分析流程，包括文本预处理、分词、
        特征提取和情感预测，最后输出结果。
        
        Args:
            text (str): 待分析的文本
        """
        # 文本预处理
        text_corpus = self.replace_text(text)
        
        # 情感预测
        result = self.predict_score(text_corpus)

        # 提取负面和正面情感概率
        neg = result[0][0]  # 负面情感概率
        pos = result[0][1]  # 正面情感概率

        # 输出分析结果
        print('差评： {} 好评： {}'.format(neg, pos))
