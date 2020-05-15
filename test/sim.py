from gensim import models

w2v = models.KeyedVectors.load_word2vec_format(
    "./models/final/word2Vec2.model", binary=True)
s1 = w2v.similarity('番茄', '西红柿')
print(s1)
s1 = w2v.similarity('土豆', '马铃薯')
print(s1)
s1 = w2v.similarity('编程', '程序设计')
print(s1)
s1 = w2v.similarity('企业', '公司')
print(s1)
s1 = w2v.similarity('亚洲', '欧洲')
print(s1)
s1 = w2v.similarity('太阳', '月亮')
print(s1)
s1 = w2v.similarity('宇宙', '银河')
print(s1)
s1 = w2v.similarity('北京', '上海')
print(s1)
s1 = w2v.similarity('彗星', '稳定')
print(s1)