# -*- coding: utf-8 -*-
"""
朴素贝叶斯情感分析模型测试脚本

使用Pipeline方式构建和测试朴素贝叶斯情感分析模型，
评估模型在测试集上的准确率表现。

Author: 3inchtime
Date: 2019
"""

import re
import csv
import random

import numpy as np
import jieba

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# 加载用户自定义词典
jieba.load_userdict('./data/userdict.txt')

# 文件路径配置
file_path = './data/review.csv'      # 影评数据文件
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

    # 将读取出来的语料转为list格式
    review_data = np.array(rows).tolist()
    
    # 打乱语料的顺序，确保数据随机性
    random.shuffle(review_data)

    review_list = []     # 存储评论文本
    sentiment_list = []  # 存储情感标签
    
    # 分离数据：第一列为差评/好评标签，第二列为评论内容
    for words in review_data:
        review_list.append(words[1])   # 评论文本
        sentiment_list.append(words[0]) # 情感标签

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
    
    对单条评论进行文本清洗、jieba分词，并过滤掉停用词，
    返回有意义的词汇列表用于后续特征提取。
    
    Args:
        review (str): 原始评论文本
        
    Returns:
        list: 分词后的词汇列表（已过滤停用词）
    """
    # 加载停用词
    stop_words = load_stopwords(stopword_path)
    
    # 文本清洗：只保留中文字符和英文字母
    review = re.sub("[^\u4e00-\u9fa5^a-z^A-Z]", '', review)
    
    # 使用jieba进行中文分词
    review = jieba.cut(review)
    
    # 过滤停用词
    if stop_words:
        all_stop_words = set(stop_words)  # 转换为集合提高查找效率
        words = [w for w in review if w not in all_stop_words]

    return words


# ==================== 主测试流程 ====================

# 加载影评语料数据
review_list, sentiment_list = load_corpus(file_path)

# 数据集划分：按照4:1的比例分为训练集和测试集
n = len(review_list) // 5  # 计算测试集大小（总数据的1/5）
train_review_list, train_sentiment_list = review_list[n:], sentiment_list[n:]  # 训练集（4/5）
test_review_list, test_sentiment_list = review_list[:n], sentiment_list[:n]    # 测试集（1/5）

# 输出数据集信息
print('训练集数量： {}'.format(str(len(train_review_list))))
print('测试集数量： {}'.format(str(len(test_review_list))))

# 文本预处理：对训练集进行分词和停用词过滤
review_train = [' '.join(review_to_text(review)) for review in train_review_list]
sentiment_train = train_sentiment_list  # 训练集对应的情感标签

# 文本预处理：对测试集进行分词和停用词过滤
review_test = [' '.join(review_to_text(review)) for review in test_review_list]
sentiment_test = test_sentiment_list    # 测试集对应的情感标签

# ==================== 特征提取器初始化 ====================

# 词频向量化器（用于比较，但在Pipeline中会重新定义）
count_vec = CountVectorizer(max_df=0.8, min_df=3)

# TF-IDF向量化器（备用）
tfidf_vec = TfidfVectorizer()


def MNB_Classifier():
    """
    构建朴素贝叶斯分类器Pipeline
    
    使用Pipeline将词频向量化和朴素贝叶斯分类器串联起来，
    实现流式化的文本分类处理流程。
    
    Returns:
        Pipeline: 包含词频向量化和朴素贝叶斯分类器的Pipeline对象
    """
    return Pipeline([
        ('count_vec', CountVectorizer()),  # 第一步：词频向量化
        ('mnb', MultinomialNB())          # 第二步：多项式朴素贝叶斯分类
    ])


# ==================== 模型训练和测试 ====================

# 创建分类器Pipeline
mnbc_clf = MNB_Classifier()

# 训练模型
mnbc_clf.fit(review_train, sentiment_train)

# 计算并输出测试集准确率
accuracy = mnbc_clf.score(review_test, sentiment_test)
print('测试集准确率： {}'.format(accuracy))

# ==================== 错误分析 ====================

# 对测试集进行预测
predictions = mnbc_clf.predict(review_test).tolist()

# 收集预测错误的样本
err_list = []
for i in range(len(review_test)):
    # 如果预测结果与真实标签不符
    if predictions[i] != sentiment_test[i]:
        error_data = {
            'true_sentiment': sentiment_test[i],    # 真实情感标签
            'predicted_sentiment': predictions[i],  # 预测情感标签
            'review': review_test[i]               # 评论文本
        }
        err_list.append(error_data)

# 可选：输出错误样本进行分析
# for err in err_list:
#     print('预测错误样本： 真实标签={}, 预测标签={}, 评论={}'.format(
#         err['true_sentiment'], err['predicted_sentiment'], err['review'][:50]))
