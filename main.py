import re

import jieba
from gensim import models, corpora
from opencc import OpenCC

import process_train_data

pattern = re.compile(u'[\u4e00-\u9fa5]+')


def pre_step(arr, cc, stop_words):
    data = []
    for word in arr:
        m = pattern.match(word)
        if m is not None:
            d = m.group()
            if d not in stop_words:
                data.append(cc.convert(d))
    return data


def cal_sim(str1, str2):
    stop_words = process_train_data.get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    cc = OpenCC("t2s")
    s1 = jieba.cut(str1, cut_all=False)
    s2 = jieba.cut(str2, cut_all=False)
    # s1、s2去重
    data1 = pre_step(s1, cc, stop_words)
    data2 = pre_step(s2, cc, stop_words)
    dictionary = corpora.Dictionary.load("./tf_idf_models/tfidf.dictionary")
    vec1 = dictionary.doc2bow(data1)
    # vec1[词的数字表示,频次]
    vec2 = dictionary.doc2bow(data2)
    print(vec1)
    print(vec2)
    new_dictionary = dictionary.token2id
    new_dictionary = {v: k for k, v in new_dictionary.items()}
    tfidf = models.TfidfModel.load("./tf_idf_models/tfidf.model")
    tf1 = tfidf[vec1]
    # tf1[词的数字表示，权重]
    tf2 = tfidf[vec2]

    w2v = models.KeyedVectors.load_word2vec_format(
        "./models/final/word2Vec2.model", binary=True)
    sim1 = 0
    total_wight1 = 0
    for t1 in tf1:
        m = 0
        word1 = new_dictionary.get(t1[0])
        for t2 in tf2:
            word2 = new_dictionary.get(t2[0])
            d = w2v.similarity(word1, word2)
            if d > m:
                m = d
        wight = t1[1]
        print(str(m) + "-------" + str(wight))
        sim1 += m * wight
        total_wight1 += wight
    sim1 = (sim1 / total_wight1)
    print("total:" + str(total_wight1) + ",sim1:" + str(sim1))
    print("################################")
    # 第二句
    sim2 = 0
    total_wight2 = 0
    for t2 in tf2:
        m = 0
        word2 = new_dictionary.get(t2[0])
        for t1 in tf1:
            word1 = new_dictionary.get(t1[0])
            d = w2v.similarity(word2, word1)
            if d > m:
                m = d
        wight = t2[1]
        print(str(m) + "-------" + str(wight))
        sim2 += m * wight
        total_wight2 += wight
    sim2 = (sim2 / total_wight2)
    print("total:" + str(total_wight1) + ",sim2:" + str(sim2))
    print("################################")
    sim = (sim1 + sim2) / 2
    print(sim)


if __name__ == "__main__":
    # cal_sim("我喜欢吃马铃薯，他也喜欢吃", "我喜欢吃马铃薯，他也喜欢吃")
    cal_sim("我喜欢吃马铃薯，他也喜欢吃", "我爱吃土豆，他也爱吃")
