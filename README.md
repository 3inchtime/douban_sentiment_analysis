# 豆瓣影评情感分析系统

基于朴素贝叶斯算法实现的中文影评情感分析工具，能够自动判断影评文本的情感倾向（正面/负面）。

## 📖 项目简介

本项目使用机器学习中的朴素贝叶斯算法，结合TF-IDF特征提取技术，构建了一个专门用于分析豆瓣影评情感的分类系统。系统能够自动识别影评文本中的情感倾向，为电影推荐、用户行为分析等应用提供支持。

### 🎯 主要特性

- **高准确率**: 在测试集上达到约80%的分类准确率
- **中文优化**: 专门针对中文文本进行优化，支持jieba分词
- **自定义词典**: 支持用户自定义词典，提高特定领域词汇的识别准确性
- **停用词过滤**: 内置停用词过滤机制，提高特征质量
- **易于使用**: 提供简洁的API接口，方便集成到其他项目中

### 📊 数据集信息

- **数据来源**: 豆瓣Top250电影排行榜中的用户影评
- **数据规模**: 约50,000条影评数据
- **数据平衡**: 正面评论和负面评论各占50%
- **数据获取**: 基于Scrapy框架爬取（[爬虫项目地址](https://github.com/3inchtime/douban_movie_review)）
- **数据划分**: 训练集:测试集 = 4:1

### 🔍 算法原理

本项目采用多项式朴素贝叶斯分类器，结合TF-IDF特征提取：

1. **文本预处理**: 去除URL、特殊字符，按标点符号分句
2. **中文分词**: 使用jieba分词器进行中文分词
3. **停用词过滤**: 过滤无意义的停用词
4. **特征提取**: 使用TF-IDF算法提取文本特征
5. **模型训练**: 使用多项式朴素贝叶斯算法训练分类器
6. **情感预测**: 输出正面和负面情感的概率

### ⚠️ 模型局限性

由于电影评论的复杂性，部分好评中可能包含负面情感词汇。例如纪录片《海豚湾》的评论：

> 我觉得大部分看本片会有感的人，都不知道，中国的白暨豚已经灭绝8年了，也不会知道，长江里的江豚也仅剩1000左右了。与其感慨，咒骂日本人如何捕杀海豚，不如做些实际的事情，保护一下长江里的江豚吧，没几年，也将绝迹了。中国人做出来的事情，也不会比小日本好到哪儿去。

这类评论虽然包含负面词汇，但整体情感倾向是正面的。通过数据清洗和特征工程优化，可以进一步提高模型准确率。

## 🚀 快速开始

### 环境要求

- Python 3.6+
- jieba
- scikit-learn
- numpy
- pandas (可选，用于数据处理)

### 安装依赖

```bash
pip install jieba scikit-learn numpy
```

### 项目结构

```
douban_sentiment_analysis/
├── data/                              # 数据文件夹
│   ├── review.csv                     # 影评数据集
│   ├── stopwords.txt                  # 停用词列表
│   ├── userdict.txt                   # 用户自定义词典
│   └── bayes.pkl                      # 训练好的模型（训练后生成）
├── native_bayes_sentiment_analyzer.py # 情感分析器核心类
├── native_bayes_train.py              # 模型训练脚本
├── native_bayes_test.py               # 模型测试脚本
├── run_test.py                        # 使用示例脚本
├── native_bayes.ipynb                 # Jupyter Notebook演示
└── README.md                          # 项目说明文档
```

## 📝 使用方法

### 1. 训练模型

首次使用需要先训练模型：

```bash
python native_bayes_train.py
```

输出示例：
```
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.570 seconds.
Prefix dict has been built succesfully.
训练集数量： 40906
测试集数量： 10226
训练完成
```

### 2. 测试模型性能

```bash
python native_bayes_test.py
```

输出示例：
```
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.572 seconds.
Prefix dict has been built succesfully.
训练集数量： 40906
测试集数量： 10226
测试集准确率： 0.8043380027112517
```

### 3. 使用情感分析器

#### 基本使用

```python
from native_bayes_sentiment_analyzer import SentimentAnalyzer

# 配置文件路径
model_path = './data/bayes.pkl'
userdict_path = './data/userdict.txt'
stopword_path = './data/stopwords.txt'

# 创建分析器实例
analyzer = SentimentAnalyzer(
    model_path=model_path,
    userdict_path=userdict_path,
    stopword_path=stopword_path
)

# 分析文本情感
text = "这部电影真的很棒，演员演技精湛，剧情引人入胜！"
analyzer.analyze(text)
# 输出: 差评： 0.123 好评： 0.877
```

#### 批量分析

```python
# 批量分析多条评论
reviews = [
    "这部电影真的很棒，演员演技精湛！",
    "剧情拖沓，演技尴尬，不推荐观看。",
    "还可以吧，中规中矩的电影。"
]

for review in reviews:
    print(f"评论: {review}")
    analyzer.analyze(review)
    print("-" * 50)
```

#### 完整示例

```python
from native_bayes_sentiment_analyzer import SentimentAnalyzer

# 配置文件路径
model_path = './data/bayes.pkl'
userdict_path = './data/userdict.txt'
stopword_path = './data/stopwords.txt'

# 创建分析器实例
analyzer = SentimentAnalyzer(
    model_path=model_path, 
    stopword_path=stopword_path, 
    userdict_path=userdict_path
)

# 测试负面评论
negative_text = ('倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。'
                '虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，'
                '但真心没想到能差到这个地步。'
                '节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。')

print("负面评论分析:")
analyzer.analyze(negative_text)
# 输出: 差评： 0.738 好评： 0.262

# 测试正面评论
positive_text = "这部电影真的很棒，演员演技精湛，剧情引人入胜，强烈推荐！"
print("\n正面评论分析:")
analyzer.analyze(positive_text)
# 输出: 差评： 0.123 好评： 0.877
```

### 4. 运行示例

项目提供了完整的使用示例：

```bash
python run_test.py
```

## 📚 API 文档

### SentimentAnalyzer 类

#### 初始化参数

- `model_path` (str): 训练好的模型文件路径
- `userdict_path` (str): 用户自定义词典文件路径
- `stopword_path` (str): 停用词文件路径

#### 主要方法

##### `analyze(text)`
分析文本情感并打印结果
- **参数**: `text` (str) - 待分析的文本
- **返回**: None（直接打印结果）

##### `predict_score(text_corpus)`
返回情感分析的概率值
- **参数**: `text_corpus` (list) - 预处理后的句子列表
- **返回**: numpy.ndarray - 包含[负面概率, 正面概率]的数组

##### `replace_text(text)`
文本预处理和清洗
- **参数**: `text` (str) - 原始文本
- **返回**: list - 分割后的句子列表

## 🔧 技术细节

### 特征工程

1. **文本预处理**
   - URL链接移除
   - 特殊字符清理
   - 中文标点符号分句

2. **分词处理**
   - 使用jieba分词器
   - 支持用户自定义词典
   - 停用词过滤

3. **特征提取**
   - CountVectorizer: 词频统计
   - TfidfTransformer: TF-IDF权重计算
   - 参数配置: max_df=0.8, min_df=3

### 模型架构

- **算法**: 多项式朴素贝叶斯 (MultinomialNB)
- **特征**: TF-IDF向量
- **分类**: 二分类（正面/负面）
- **评估指标**: 准确率 (Accuracy)

### 性能优化

- 使用Pipeline进行流水线处理
- 停用词集合化提高查找效率
- 模型序列化存储，避免重复训练

## 📁 数据文件说明

### review.csv
影评数据集，格式：`情感标签,评论内容`
- 0: 负面评论
- 1: 正面评论

### stopwords.txt
停用词列表，每行一个停用词

### userdict.txt
用户自定义词典，包含影评领域的专业词汇

### bayes.pkl
训练好的模型文件，包含：
- clf: 朴素贝叶斯分类器
- vectorizer: 词频向量化器
- tfidftransformer: TF-IDF转换器

## 🎯 应用场景

- **电影推荐系统**: 基于用户评论情感进行个性化推荐
- **舆情监控**: 监控电影上映后的用户反馈
- **内容审核**: 自动识别负面评论进行人工审核
- **市场分析**: 分析电影市场反响和用户偏好
- **评分预测**: 结合情感分析预测电影评分

## 🚧 改进方向

1. **模型优化**
   - 尝试深度学习模型（LSTM、BERT等）
   - 集成多种算法提高准确率
   - 增加情感强度分析

2. **特征工程**
   - 添加词性标注特征
   - 考虑句法依存关系
   - 引入情感词典

3. **数据扩充**
   - 增加更多电影类型的评论
   - 处理数据不平衡问题
   - 添加细粒度情感标签

## 📖 参考资料

- [详细技术博客](https://juejin.cn/post/6844903941226921991)
- [豆瓣影评爬虫项目](https://github.com/3inchtime/douban_movie_review)
- [scikit-learn 官方文档](https://scikit-learn.org/)
- [jieba 分词工具](https://github.com/fxsjy/jieba)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👨‍💻 作者

- **3inchtime** - *初始工作* - [GitHub](https://github.com/3inchtime)

## 🙏 致谢

- 感谢豆瓣网提供的丰富影评数据
- 感谢开源社区提供的优秀工具和库
- 感谢所有为这个项目做出贡献的开发者
