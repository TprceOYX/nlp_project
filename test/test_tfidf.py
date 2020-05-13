from gensim import models, corpora
tfidf = models.TfidfModel.load("./tf_idf_models/tfidf.model")
print(type(tfidf))
dictionary = corpora.Dictionary.load("./tf_idf_models/tfidf.dictionary")
print(type(dictionary))
