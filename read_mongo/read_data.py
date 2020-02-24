'''
用于读取数据库中百度百科词条数据，使用jieba分词后保存为txt文件
豆瓣短评和长评，使用另外的函数
'''
import codecs
import re

import jieba.analyse
import pymongo

pattern = re.compile(u'[\u4e00-\u9fa5]+')


def cut_text1(text, stop_words, converter):
    words = jieba.cut(text["text"], cut_all=False)
    line = ""
    for word in words:
        m = pattern.match(word)
        if m is not None:
            d = m.group()
            if d not in stop_words:
                line += converter.convert(d)
                line += " "
    return line


def cut_text2(text, stop_words, converter):
    target = codecs.open("./cache/" + str(text["_id"]) + ".txt",
                         'w',
                         encoding="utf8")
    words = jieba.cut(text["text"], cut_all=False)
    line = ""
    for word in words:
        m = pattern.match(word)
        if m is not None:
            d = m.group()
            if d not in stop_words:
                line += converter.convert(d)
                line += " "
    target.writelines(line)
    target.close()


def cut_wiki(text, stop_words):
    words = jieba.cut(text, cut_all=False)
    line = ""
    count = 0
    for word in words:
        m = pattern.match(word)
        if m is not None:
            d = m.group()
            if d not in stop_words:
                count += 1
                line += d
                line += " "
    if count <= 5:
        return ""
    return line


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
        f.close()
    return stopwords


def process_data(stop_words,
                 collection,
                 converter,
                 save_file_path,
                 start=0,
                 batch=1000):
    result = collection.find({}, {
        "_id": 1,
        "text": 1
    }).skip(start).limit(batch)
    # 加载停用词，如标点符号等
    # jieba.enable_parallel(multiprocessing.cpu_count())
    count = 0
    # [\u0800-\u4e00]+
    output = codecs.open(save_file_path, "w+", encoding="utf8")
    for x in result:
        text = cut_text1(x, stop_words, converter)
        count += 1
        if count % 100 == 0:
            print("process ", count)
        if text != "":
            output.write(text + "\n")
    output.close()


def process_wiki(read_file_path, save_file_path, stop_words):
    file = codecs.open(read_file_path, "r", encoding="utf-8")
    # 写文件
    output = codecs.open(save_file_path, "w+", encoding="utf-8")
    for line in file:
        line = line.strip('\n')
        text = cut_wiki(line, stop_words)
        if text != "":
            output.write(text + "\n")
    output.close()


if __name__ == "__main__":
    client = pymongo.MongoClient(
        "mongodb://tprce:1634834938@47.94.0.240:27017/")
    db = client["document"]
    collection = db["baike"]
    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    process_data(stop_words, collection, 30000)
