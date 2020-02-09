from gensim import models

model = models.Word2Vec.load("models/word2Vec.model")
# 输入一个词找出相似的前10个词
one_corpus = ["东京"]
result = model.most_similar(one_corpus[0], topn=10)
print(result)
