from gensim import models

model = models.Word2Vec.load("./models/train/word2vec.model")
model.wv.wv.save_word2vec_format("./models/final/word2Vec2.model", binary=True)
