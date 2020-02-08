'''
用于读取数据库数据，使用jieba分词后保存为txt文件
'''
import pymongo
import codecs
import jieba.analyse


def cut_text(text, stop_words):
    target = codecs.open("./train_data/" + str(text["_id"]) + ".txt",
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


def process_data():
    # 连接数据库
    myclient = pymongo.MongoClient(
        "mongodb://tprce:1634834938@47.94.0.240:27017/")
    mydb = myclient["document"]
    mycol = mydb["baike"]
    myresult = mycol.find({}, {"_id": 1, "text": 1}).limit(1000)
    # 加载停用词，如标点符号等
    stop_file = codecs.open("./data/stop_words.txt", "r", encoding="utf8")
    stop_words = stop_file.read().split("\n")
    jieba.set_dictionary("./data/dict.txt.big")
    for x in myresult:
        cut_text(x, stop_words)


if __name__ == "__main__":
    process_data()
