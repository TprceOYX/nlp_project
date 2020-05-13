# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import sklearn.manifold as ts
# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from gensim import models
model = models.KeyedVectors.load_word2vec_format(
    "./models/final/word2Vec2.model", binary=True)

# y1 = model.similarity('国家', '国务院') #算两个词的相似度/相关程度
# print('两个词的相似度: ', y1)
#
# y2 = model.most_similar('自动化', topn=10) #计算某个词的最相关词列表
# print('相关词有：\n', y2)
#
# y3 = model.doesnt_match('书 书籍 教材 课本 文具 水果'.split()) #寻找不合群的词
# print('不合群的词有：', y3)
#
# y4 = model['计算机'] #直接获取某个单词的向量表示
# print(y4)
#
# y5 = model.similar_by_word('清华大学', topn=5)
# print(y5)
#
y6 = model.vector_size  # 词向量维度
print('词向量维度:', y6)

# 模型可视化
# 使用t-SNE可视化学习的嵌入。t-SNE是一种数据可视化工具，可将数据的维度降至2或3维，从而可以轻松进行绘制。
# 由于t-SNE算法的空间复杂度是二次的，因此在本教程中，我们将仅查看模型的一部分。

# 我们使用下面的代码从我们的词汇中选择10,000个单词
count = len(model.vectors)
print(model.vocab)
# count = 40000
word_vectors_matrix = np.ndarray(shape=(count, 256), dtype='float32')
word_list = []
i = 0
for word in model.vocab:
    word_vectors_matrix[i] = model[word]
    word_list.append(word)
    i = i + 1
    if i == count:
        break
print("word_vectors_matrix shape is: ", word_vectors_matrix.shape)

# 由于模型是一个300维向量，利用Scikit-Learn 中的降维算法t-SNE
# 初始化模型并将我们的单词向量压缩到二维空间
# 降维操作

tsne = ts.TSNE(n_components=2, random_state=0)
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
print("word_vectors_matrix_2d shape is: ", word_vectors_matrix_2d.shape)

# 数据框，其中包含所选单词和每个单词的x和y坐标

points = pd.DataFrame(
    [(word, coords[0], coords[1])
     for word, coords in [(word, word_vectors_matrix_2d[word_list.index(word)])
                          for word in word_list]],
    columns=["word", "x", "y"])
print("Points DataFrame built")
print(points.head(10))

# DataFrame来绘制我们的单词向量

# 方法一
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题,否则会显示成方块
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# #方法二
# myfont = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simsunb.ttf')
# mpl.rcParams['axes.unicode_minus'] = False
# plt.title(u'标题', fontproperties=myfont)
# 四种预设，按相对尺寸的顺序(线条越来越粗)，分别是paper，notebook, talk, and poster
sns.set_context('poster')
# points.plot.scatter("x", "y", s=10, figsize=(20, 12))
# plt.show()

# 放大到一些区域，以便看到单词的相似性。我们创建一个函数，
# 创建x和y坐标的边界框，并只绘制该边界框之间的单词。


def plot_region(x_bounds, y_bounds):
    slice = points[(x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
                   (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]

    ax = slice.plot.scatter("x", "y", s=10, figsize=(20, 12))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.05, point.y + 0.05, point.word,
                fontsize=11)  # text可以将文本绘制在图表指定坐标(x,y)


plot_region(x_bounds=(-35, -20), y_bounds=(-25, -10))
plt.show()
