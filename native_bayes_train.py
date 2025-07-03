# -*- coding: utf-8 -*-
"""
朴素贝叶斯情感分析模型训练脚本

使用豆瓣影评数据训练朴素贝叶斯分类器，用于中文文本情感分析。
训练过程包括数据加载、文本预处理、特征提取、模型训练和保存。

Author: 3inchtime
Date: 2019
"""

import os
import csv
import random
import pickle

import numpy as np
import jieba

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# 加载用户自定义词典，提高分词准确性
jieba.load_userdict("./data/userdict.txt")

# 文件路径配置
file_path = './data/review.csv'  # 影评数据文件
model_export_path = './data/bayes.pkl'  # 模型保存路径
stopword_path = './data/stopwords.txt'  # 停用词文件


def load_corpus(corpus_path):
    """
    加载影评语料数据
    
    从CSV文件中读取影评数据，包含情感标签和评论内容。
    数据格式：第一列为情感标签（0=负面，1=正面），第二列为评论文本。
    
    Args:
        corpus_path (str): 语料文件路径
        
    Returns:
        tuple: (评论列表, 情感标签列表)
    """
    with open(corpus_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    # 转换为列表格式
    review_data = np.array(rows).tolist()
    
    # 打乱数据顺序，避免数据偏差
    random.shuffle(review_data)

    review_list = []  # 存储评论文本
    sentiment_list = []  # 存储情感标签
    
    # 分离评论文本和情感标签
    for words in review_data:
        review_list.append(words[1])  # 评论文本
        sentiment_list.append(words[0])  # 情感标签

    return review_list, sentiment_list


def load_stopwords(file_path):
    """
    加载停用词列表
    
    从文件中读取停用词，用于在分词过程中过滤无意义的词汇。
    
    Args:
        file_path (str): 停用词文件路径
        
    Returns:
        list: 停用词列表
    """
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
       stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words


def review_to_text(review):
    """
    评论文本预处理和分词
    
    对单条评论进行jieba分词，并过滤掉停用词，
    返回有意义的词汇列表用于后续特征提取。
    
    Args:
        review (str): 原始评论文本
        
    Returns:
        list: 分词后的词汇列表（已过滤停用词）
    """
    # 加载停用词
    stop_words = load_stopwords(stopword_path)
    
    # 使用jieba进行中文分词
    review = jieba.cut(review)
    
    # 转换为集合以提高查找效率
    all_stop_words = set(stop_words)
    
    # 过滤停用词，保留有意义的词汇
    review_words = [w for w in review if w not in all_stop_words]

    return review_words


# ==================== 主训练流程 ====================

# 加载语料数据
review_list, sentiment_list = load_corpus(file_path)

# 数据集划分：按照4:1的比例划分训练集和测试集
n = len(review_list) // 5  # 计算测试集大小（总数据的1/5）

# 分割数据集
train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]  # 训练集（4/5）
test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]    # 测试集（1/5）

# 输出数据集信息
print('训练集数量： {}'.format(str(len(train_review_list))))
print('测试集数量： {}'.format(str(len(test_review_list))))

# 文本预处理：对训练集进行分词和停用词过滤
review_train = [' '.join(review_to_text(review)) for review in train_review_list]
sentiment_train = train_sentiment_list

# 文本预处理：对测试集进行分词和停用词过滤
review_test = [' '.join(review_to_text(review)) for review in test_review_list]
sentiment_test = test_sentiment_list

# ==================== 特征提取 ====================

# 初始化词频向量化器
# max_df=0.8: 忽略出现在超过80%文档中的词汇（过于常见）
# min_df=3: 忽略出现次数少于3次的词汇（过于稀少）
vectorizer = CountVectorizer(max_df=0.8, min_df=3)

# 初始化TF-IDF转换器
tfidftransformer = TfidfTransformer()

# ==================== 模型训练 ====================

# 特征提取：先转换成词频矩阵，再计算TF-IDF值
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(review_train))

# 训练朴素贝叶斯分类器（多项式分布）
clf = MultinomialNB().fit(tfidf, sentiment_train)

# ==================== 模型保存 ====================

# 将训练好的模型组件保存到pickle文件
with open(model_export_path, 'wb') as file:
    model_dict = {
        "clf": clf,                           # 训练好的分类器
        "vectorizer": vectorizer,             # 词频向量化器
        "tfidftransformer": tfidftransformer, # TF-IDF转换器
    }
    pickle.dump(model_dict, file)

print("训练完成")
