import codecs
import os
import re
import sys

import jieba
import numpy as np
from gensim import corpora, models
from opencc import OpenCC

pattern = re.compile(u'[\u4e00-\u9fa5]+')


def string_hash(source):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7
        m = 1000003
        mask = 2**128 - 1
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
        x ^= len(source)
        if x == -1:
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]
        return str(x)


def pre_step(arr, cc, stop_words):
    data = []
    for word in arr:
        m = pattern.match(word)
        if m is not None:
            d = m.group()
            if d not in stop_words:
                data.append(cc.convert(d))
    return data


def distance_haiming(hash1, hash2):
    t1 = '0b' + str(hash1)
    t2 = '0b' + str(hash2)
    n = int(t1, 2) ^ int(t2, 2)
    i = 0
    while n:
        n &= (n - 1)
        i += 1
    return i


def simhash(tf1, tf2, new_dictionary):
    keyList1 = []
    for t in tf1:
        feature = string_hash(new_dictionary.get(t[0]))
        temp = []
        for i in feature:
            if i == '1':
                temp.append(int(t[1] * 10))
            else:
                temp.append(-int(t[1] * 10))
        print(new_dictionary.get(t[0]) + "->" + str(temp))
        keyList1.append(temp)
    key1 = np.sum(np.array(keyList1), axis=0)
    print("s1 key -> " + str(key1))
    print("-----------------------------------------------")
    keyList2 = []
    for t in tf2:
        feature = string_hash(new_dictionary.get(t[0]))
        temp = []
        for i in feature:
            if i == '1':
                temp.append(int(t[1] * 10))
            else:
                temp.append(-int(t[1] * 10))
        print(new_dictionary.get(t[0]) + "->" + str(temp))
        keyList2.append(temp)
    key2 = np.sum(np.array(keyList2), axis=0)
    print("s2 key -> " + str(key2))
    print("-----------------------------------------------")
    if (keyList1 == [] or keyList2 == []):  # 编码读不出来
        return
    simhash1 = ''
    for i in key1:
        if (i > 0):
            simhash1 = simhash1 + '1'
        else:
            simhash1 = simhash1 + '0'
    simhash2 = ''
    for i in key2:
        if (i > 0):
            simhash2 = simhash2 + '1'
        else:
            simhash2 = simhash2 + '0'
    print("simhash1 = " + simhash1)
    print("simhash2 = " + simhash2)
    distance = distance_haiming(simhash1, simhash2)
    print("distance = " + str(distance))


def read_file(path):
    file = codecs.open(path, "r", encoding="utf-8")
    data = []
    for line in file:
        data.append(line.strip())
    return "".join(data)


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
    count = len(sys.argv)
    sim = 0
    if count != 3:
        # cal_sim("我喜欢吃马铃薯，他也喜欢吃", "我喜欢吃马铃薯，他也喜欢吃")
        # cal_sim("我喜欢吃马铃薯，他也喜欢吃", "我爱吃土豆，他也爱吃")
        str1 = """黎明，一丝微弱的阳光从玻璃窗里透了进来，如同一丝金光，使屋里不再黑暗。
                东方一点儿一点儿泛着鱼肚色的天空，飘着五颜六色的朝霞，有：降紫的、金黄的、青色的……甚至还有一些火红色的火烧云，好像把大半个天给点燃，简直就是美不胜收。
                今天的朝霞十分奇异，既不像棉花糖，又不像绵羊，而像鱼鳞，很罕见，如果把蓝天比作大海，那这朝霞就是海浪，令人陶醉。在东南方向，还有一道七色彩虹，像桥一样，
                也许是太阳公公的桥梁吧！太阳仿佛是一个胆小的姑娘，不敢往上爬升，只是在地平线上露出了小脑袋。天由鱼肚色慢慢变成似大海的蔚蓝色，太阳爬得非常缓慢，
                未免令人着急，渐渐地，太阳胆子变大了，露出了半个头，更加激动人心，接着，已经露出了三分之二。很快，太阳不断地努力，终于爬上了天空，
                此时的太阳不像大火球，因为它要歇一歇，阳光十分温和。阳光把河水照得仿佛金子一样闪闪发光，鱼儿们游来游去，小鸟纷纷“叽叽喳喳”地唱歌。就这样，大地苏醒过来了。"""
        str2 = """早上，一丝微弱的阳光从玻璃窗里透了进来，好像一丝金光，使屋里不再黑暗。
                东方一点一点泛着鱼肚色的天空，飘着五颜六色的朝霞，有：降紫的、金黄的、青色的……甚至还有一些火红色的火烧云，好像把大半个天给点燃，简直就是美不胜收。
                今天的朝霞十分奇异，既不像棉花糖，又不像绵羊，而像鱼鳞，很罕见，如果把蓝天比作大海，那这朝霞就是海浪，令人陶醉。在东南方向，还有一道七色彩虹，像桥一样，
                也许是太阳公公的桥梁吧！太阳仿佛是一个胆小的姑娘，不敢往上爬升，只是在地平线上露出了小脑袋。天由鱼肚色慢慢变成似海洋的蔚蓝色，太阳爬得非常缓慢，
                未免令人着急，渐渐地，太阳胆子变大了，露出了半个头，更加激动人心，接着，已经露出了三分之二。很快，太阳不断地努力，终于爬上了天空，
                此时的太阳不像大火球，因为它要歇一歇，阳光十分温和。阳光把河水照得仿佛金子一样闪闪发光，鱼儿们游来游去，小鸟纷纷“叽叽喳喳”地唱歌。就这样，大地苏醒过来了。"""
        sim = cal_sim(str1, str2, stop_words, w2v, dictionary, new_dictionary,
                      tfidf)
    else:
        a1 = sys.argv[1]
        a2 = sys.argv[2]
        if os.path.exists(a1) and os.path.exists(a2):
            sim = cal_sim(read_file(a1), read_file(a2), stop_words, w2v,
                          dictionary, new_dictionary, tfidf)
        else:
            sim = cal_sim(a1, a2, "test")
    if sim > 0.80000:
        print("文本重复")
    else:
        print("文本不重复")
    # s = string_hash("我")
    # print(s)
