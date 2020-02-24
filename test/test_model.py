from gensim import models

w2v = models.KeyedVectors.load_word2vec_format("./models/final/word2Vec2.model", binary=True)
# 输入一个词找出相似的前10个词
one_corpus = ["番茄"]
result = w2v.most_similar(one_corpus[0], topn=10)
print(result)
