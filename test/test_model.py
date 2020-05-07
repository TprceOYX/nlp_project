from gensim import models
import matplotlib.pyplot as plt
import sklearn.manifold as ts
from pylab import mpl

w2v = models.KeyedVectors.load_word2vec_format(
    "./models/final/word2Vec2.model", binary=True)
print(len(w2v.vectors))
# 输入一个词找出相似的前10个词
one_corpus = ["番茄", "亚洲", "编程", "企业", "地理"]
for item in one_corpus:
    result = w2v.most_similar(item, topn=10)
    print("与" + item + "最相似的10个词为：")
    print(result)

random_word = [
    '中国', '我国', '中华', '日本', '亚洲', '非洲', '韩国', '互联网', '因特网', '人工智能', '高科技',
    '编程', '程序设计', '面向对象', '蔬菜', '水果', '肉类', '苹果', '西红柿', '民国', '四川', '世界',
    '牛顿', '科学家', '国家电网', '菠萝', '火龙果', '橘子', '西瓜', '鼠', '牛', '虎', '兔', '龙', '蛇',
    '马', '羊', '猴', '鸡', '狗', '猪', '动物', '俄罗斯', '德国', '法国', '意大利', '巴黎', '罗马',
    '北京', '上海', '广州', '深圳', '企业', '音乐', '艺术', '美术', '电影', '动画', '猫', '艳丽',
    '飘逸', '飘扬', '容态', '容颜', '姿态', '蒙古族', '内蒙古', '蒙古', '美国', '加州', '纽约', '华盛顿',
    '国家', '经营', '资格', '专业', '资质', '财务状况', '技术', '能力', '管理', '业绩', '信誉', '信用',
    '征收', '进口', '关税', '证券', '经济', '交易', '公民', '推广', '计费'
]
# 降维
X_tsne = ts.TSNE(n_components=2,
                 learning_rate=100).fit_transform(w2v[random_word])
plt.figure(figsize=(14, 8))
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题,否则会显示成方块
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for i in range(len(X_tsne)):
    x = X_tsne[i][0]
    y = X_tsne[i][1]
    plt.text(x, y, random_word[i], size=16)

plt.show()
