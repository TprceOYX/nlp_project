import codecs
import os
import shutil
import tarfile

import jieba
import pymongo
from opencc import OpenCC

from read_mongo import read_data


def make_targz(output_filename, source_dir):
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
            tar.close()
        return True
    except Exception as e:
        print(e)
        return False


def clear_dir():
    shutil.rmtree("./cache")
    os.mkdir("./cache")


def get_stop_words(path):
    stopwords = []
    with codecs.open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
        f.close()
    return stopwords


def read_baike():
    cc = OpenCC("t2s")
    client = pymongo.MongoClient(host="mongodb://127.0.0.1:27017/",
                                 unicode_decode_error_handler='ignore')
    db = client["document"]
    collection = db["baike"]
    baike_total_count = collection.count()

    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    count = 1
    batch = 10000
    i = batch * (count - 1)
    while i < baike_total_count:
        read_data.process_data(stop_words, collection, cc,
                               "./train_data/baike_" + str(count), i, batch)
        # train.main_func()
        # make_targz("./train_data/baike" + str(count) + ".tar.gz", "./cache")
        # clear_dir()
        i += batch
        count += 1


def read_douban_review():
    cc = OpenCC("t2s")
    client = pymongo.MongoClient(host="mongodb://127.0.0.1:27017/",
                                 unicode_decode_error_handler='ignore')
    db = client["document"]
    collection = db["douban_review"]
    total_count = collection.count()

    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    count = 2
    batch = 30000
    i = batch * (count - 1)
    while i < total_count:
        read_data.process_data(stop_words, collection, cc,
                               "./train_data/douban_review_" + str(count), i,
                               batch)
        # train.main_func()
        # make_targz("./train_data/douban_review" + str(count) + ".tar.gz",
        #            "./cache")
        # clear_dir()
        i += batch
        count += 1


def read_douban_short():
    cc = OpenCC("t2s")
    client = pymongo.MongoClient(host="mongodb://127.0.0.1:27017/",
                                 unicode_decode_error_handler='ignore')
    db = client["document"]
    collection = db["douban_short"]
    total_count = collection.count()

    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    count = 1
    batch = 630000
    i = batch * (count - 1)
    while i < total_count:
        read_data.process_data(stop_words, collection, cc,
                               "./train_data/douban_short_" + str(count), i,
                               batch)
        # train.main_func()
        # make_targz("./train_data/douban_short" + str(count) + ".tar.gz",
        #            "./cache")
        # clear_dir()
        i += batch
        count += 1


def read_zhwiki():
    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    filenames = os.listdir("./zhwiki")
    for filename in filenames:
        print(filename)
        read_data.process_wiki("./zhwiki/" + filename,
                               "./train_data/" + filename + "_train",
                               stop_words)


def read_novels():
    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    read_data.process_novels("./novels", stop_words)


if __name__ == "__main__":
    # read_baike()
    # read_douban_review()
    # read_douban_short()
    # read_zhwiki()
    read_novels()
