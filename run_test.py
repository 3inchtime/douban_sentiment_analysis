# -*- coding: utf-8 -*-
"""
情感分析器使用示例

演示如何使用训练好的朴素贝叶斯情感分析器对影评文本进行情感分析。
这个脚本展示了完整的使用流程，包括模型加载和文本分析。

Author: 3inchtime
Date: 2019
"""

from native_bayes_sentiment_analyzer import SentimentAnalyzer

# ==================== 配置文件路径 ====================

model_path = './data/bayes.pkl'        # 训练好的模型文件路径
userdict_path = './data/userdict.txt'  # 用户自定义词典路径
stopword_path = './data/stopwords.txt' # 停用词文件路径
corpus_path = './data/review.csv'      # 原始语料文件路径（备用）

# ==================== 初始化情感分析器 ====================

# 创建情感分析器实例
analyzer = SentimentAnalyzer(
    model_path=model_path, 
    stopword_path=stopword_path, 
    userdict_path=userdict_path
)

# ==================== 情感分析示例 ====================

# 测试文本：一条负面的电影评论
test_text = ('倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。'
            '虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，'
            '但真心没想到能差到这个地步。'
            '节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。')

# 执行情感分析
print("正在分析文本情感...")
print(f"待分析文本: {test_text}")
print("\n分析结果:")
analyzer.analyze(text=test_text)
