'''
用于读取数据库中百度百科词条数据，使用jieba分词后保存为txt文件
豆瓣短评和长评，使用另外的函数
'''
import codecs

import jieba.analyse
import pymongo


def cut_text(text, stop_words):
    target = codecs.open("./cache/" + str(text["_id"]) + ".txt",
                         'w',
                         encoding="utf8")
    words = jieba.cut(text["text"], cut_all=False)
    line = ""
    # 去除标点符号
    for word in words:
        if word not in stop_words:
            line += word
            line += " "
    target.writelines(line)
    target.close()


def get_stop_words(path):
    '''
    加载停用词
    :param path:
    :return:
    '''
    stopwords = []
    with codecs.open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def process_data(stop_words, collection, start=0, batch=1000):
    result = collection.find({}, {"_id": 1, "text": 1}).skip(start).limit(batch)
    # 加载停用词，如标点符号等
    # jieba.enable_parallel(multiprocessing.cpu_count())
    count = 0
    for x in result:
        cut_text(x, stop_words)
        count += 1
        if count % 100 == 0:
            print("process ", count)


if __name__ == "__main__":
    client = pymongo.MongoClient(
        "mongodb://tprce:1634834938@47.94.0.240:27017/")
    db = client["document"]
    collection = db["baike"]
    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    process_data(stop_words, collection, 1000)
