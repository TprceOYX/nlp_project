from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('wiki_zh_jian_text.model')

testwords = ['动画', '插画', '凉宫', '设定', '喜欢']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print(testwords[i])
    print(res)
