# douban_sentiment_analysis
基于朴素贝叶斯实现的豆瓣影评情感分析

详细流程请看 <https://juejin.cn/post/6844903941226921991>

语料来自与豆瓣Top250排行榜中的影评，基于Scrapy抓取，大约5w条影评，好评差评各占50%。

豆瓣影评爬虫 <https://github.com/3inchtime/douban_movie_review>

训练集与测试集4:1，结果准确率约为80%-79%之间。

因为电影评论中有很大一部分好评中会有负面情感的词语，例如在纪录片《海豚湾》

> 我觉得大部分看本片会有感的人，都不知道，中国的白暨豚已经灭绝8年了，也不会知道，长江里的江豚也仅剩1000左右了。与其感慨，咒骂日本人如何捕杀海豚，不如做些实际的事情，保护一下长江里的江豚吧，没几年，也将绝迹了。中国人做出来的事情，也不会比小日本好到哪儿去。

所以说如果将这种类似的好评去除，则可以提高准确率。

```bash
// 测试训练
python native_bayes_test.py

Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.572 seconds.
Prefix dict has been built succesfully.
训练集数量： 40906
测试集数量： 10226
测试集准确率： 0.8043380027112517

// 训练模型
python native_bayes_train.py

Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.570 seconds.
Prefix dict has been built succesfully.
训练集数量： 40906
测试集数量： 10226
训练完成

``` 

#### Example

```Python
from native_bayes_sentiment_analyzer import SentimentAnalyzer

model_path = './data/bayes.pkl'
userdict_path = './data/userdict.txt'
stopword_path = './data/stopwords.txt'
corpus_path = './data/review.csv'

text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'

# 初始化分析器实例
analyzer = SentimentAnalyzer(model_path=model_path, stopword_path=stopword_path, userdict_path=userdict_path)

analyzer.analyze(text=text)

## 好评： 0.738 差评： 0.262

```
