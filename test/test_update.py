from gensim.models import word2vec
w2vModel = word2vec.Word2Vec.load("./models/word2Vec.model")
w2vModel.train([["hello", "world"]],
               total_examples=1,
               epochs=1)

w2vModel.save("./models/word2Vec.model")
