from gensim import models

w2v = models.KeyedVectors.load_word2vec_format("./models/final/word2Vec2.model", binary=True)
# 输入一个词找出相似的前10个词
one_corpus = ["番茄", "亚洲", "编程", "企业", "地理"]
result = w2v.most_similar(one_corpus[0], topn=10)
print(result)
result = w2v.most_similar(one_corpus[1], topn=10)
print(result)
result = w2v.most_similar(one_corpus[2], topn=10)
print(result)
result = w2v.most_similar(one_corpus[3], topn=10)
print(result)
result = w2v.most_similar(one_corpus[4], topn=10)
print(result)
