import codecs
import os
import shutil
import tarfile

import jieba
import pymongo

from read_mongo import read_baike_data


def make_targz(output_filename, source_dir):
    """
    一次性打包目录为tar.gz
    :param output_filename: 压缩文件名
    :param source_dir: 需要打包的目录
    :return: bool
    """
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False


def clear_dir():
    shutil.rmtree("./cache")
    os.mkdir("./cache")


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


def Main():
    client = pymongo.MongoClient(
        "mongodb://tprce:1634834938@47.94.0.240:27017/")
    db = client["document"]
    collection = db["baike"]
    baike_total_count = collection.count()

    stop_words = get_stop_words("./data/stop_words.txt")
    jieba.set_dictionary("./data/dict.txt.big")
    i = 0
    count = 1
    while i < baike_total_count:
        read_baike_data.process_data(stop_words, collection, i, 5000)
        # train.main_func()
        make_targz("./train_data/baike"+str(count)+".tar.gz", "./cache")
        clear_dir()
        i += 5000
        count += 1


if __name__ == "__main__":
    Main()
