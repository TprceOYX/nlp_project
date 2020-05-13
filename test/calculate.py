import codecs
import os
import re

import jieba
from gensim import corpora, models
from opencc import OpenCC

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


def get_stop_words(path):
    stopwords = []
    with codecs.open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
        f.close()
    return stopwords


def cal_sim(str1, str2, stop_words, w2v, dictionary, new_dictionary, tfidf):
    cc = OpenCC("t2s")
    s1 = jieba.cut(str1, cut_all=False)
    s2 = jieba.cut(str2, cut_all=False)
    # s1、s2去重
    data1 = pre_step(s1, cc, stop_words)
    data2 = pre_step(s2, cc, stop_words)
    a1 = []
    a2 = []
    for item in data1:
        if (item not in dictionary.token2id) or (item not in w2v.vocab):
            a1.append(item)
    for item in data2:
        if (item not in dictionary.token2id) or (item not in w2v.vocab):
            a2.append(item)
    data1 = [i for i in data1 if i not in a1]
    data2 = [i for i in data2 if i not in a2]
    vec1 = dictionary.doc2bow(data1)
    # vec1[词的数字表示,频次]
    vec2 = dictionary.doc2bow(data2)

    tf1 = tfidf[vec1]
    # tf1[词的数字表示，权重]
    tf2 = tfidf[vec2]
    return word2vec(tf1, tf2, data1, data2, new_dictionary, w2v)
    # simhash(tf1, tf2, new_dictionary)


def word2vec(tf1, tf2, s1, s2, new_dictionary, w2v):
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
        # print(str(m) + "-------" + str(wight))
        sim1 += m * wight
        total_wight1 += wight
    sim1 = (sim1 / total_wight1)
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
        # print(str(m) + "-------" + str(wight))
        sim2 += m * wight
        total_wight2 += wight
    sim2 = (sim2 / total_wight2)
    sim = (sim1 + sim2) / 2
    print(sim)
    return sim


def read_file(path):
    file = codecs.open(path, "r", encoding="utf-8")
    data = []
    for line in file:
        data.append(line.strip())
    return "".join(data)


if __name__ == "__main__":
    jieba.set_dictionary("./data/dict.txt.big")
    stop_words = get_stop_words("./data/stop_words.txt")
    w2v = models.KeyedVectors.load_word2vec_format(
        "./models/final/word2Vec2.model", binary=True)
    print("finish loading w2v model")
    dictionary = corpora.Dictionary.load("./tf_idf_models/tfidf.dictionary")
    new_dictionary = dictionary.token2id
    new_dictionary = {v: k for k, v in new_dictionary.items()}
    tfidf = models.TfidfModel.load("./tf_idf_models/tfidf.model")
    root_dir = './experiment_data'
    dirs = os.listdir(root_dir)
    dic = {}
    for dir_name in dirs:
        pairs = os.listdir(os.path.join(root_dir, dir_name))
        print(pairs)
        count = 0
        for pair in pairs:
            fileNames = os.listdir(os.path.join(root_dir, dir_name, pair))
            if len(fileNames) >= 2:
                a1 = os.path.join(root_dir, dir_name, pair, fileNames[0])
                a2 = os.path.join(root_dir, dir_name, pair, fileNames[1])
                sim = cal_sim(read_file(a1), read_file(a2), stop_words, w2v,
                              dictionary, new_dictionary, tfidf)
                print(pair + "  " + str(sim))
                if (pair.split('-')[1] == '相似'
                        and sim >= 0.8) or (pair.split('-')[1] == '不相似'
                                            and sim < 0.8):
                    count += 1
        dic[dir_name] = count / len(pairs)
    w2v.wv.save_word2vec_format("./models/final/word2Vec2.model", binary=True)
    w2v.save("./models/train/word2Vec.model")
    for key in dic.keys():
        print('%s的准确率为%f' % (key, dic[key] * 100))
