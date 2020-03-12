import codecs
import logging
import os

from gensim import corpora, models


class MyDictionary(object):
    def __init__(self, dirname):
        print("MyDictionary")
        self.dirname = dirname

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            with codecs.open(os.path.join(self.dirname, filename),
                             "r",
                             encoding="utf8") as f:
                for line in f:
                    yield line.split()
                print("MyDictionary---" + filename)
                f.close()


class MyCorpus(object):
    def __init__(self, dictionary, dirname):
        print('MyCorpus')
        self.dirname = dirname
        self.dictionary = dictionary

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            with codecs.open(os.path.join(self.dirname, filename),
                             "r",
                             encoding="utf8") as f:
                for line in f:
                    yield self.dictionary.doc2bow(line.split())
                print("MyCorpus---" + filename)
                f.close()


def make_corpus(data_path):
    filenames = os.listdir(data_path)
    document = []
    for filename in filenames:
        with codecs.open(os.path.join(data_path, filename),
                         "r",
                         encoding="utf8") as f:
            line = f.readline()
            while line:
                line = line.strip()
                if line != "" and line != " ":
                    document.append(line.split())
                line = f.readline()
            f.close()
    return document


def create_dictionary(dirname):
    dictionary = MyDictionary(dirname)
    return dictionary


def create_corpus(dirname, dictionary):
    corpus = MyCorpus(dictionary, dirname)
    return corpus


def make_tf_idf_model(model_path, data_path, model_name, dictionary_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(model_path, dictionary_name)):
        print("load tf-idf model")
        dictionary = corpora.Dictionary.load(
            os.path.join(model_path, dictionary_name))
        dictionary.add_documents(create_dictionary(data_path), prune_at=6666666)
    else:
        dictionary = corpora.Dictionary(create_dictionary(data_path),
                                        prune_at=6666666)
    print("finish create dictionary")
    # corpus = create_corpus(data_path, dictionary)
    print("finish create corpus")
    # # print(new_corpus)
    # from gensim import models
    tfidf = models.TfidfModel(dictionary=dictionary)
    tfidf.save(os.path.join(model_path, "tfidf.model"))
    dictionary.save(os.path.join(model_path, "tfidf.dictionary"))


def load_tf_idf_model(model_path, sentence):
    dictionary = corpora.Dictionary.load(
        os.path.join(model_path, "tfidf.dictionary"))
    new_dictionary = dictionary.token2id
    new_dictionary = {v: k for k, v in new_dictionary.items()}
    tfidf = models.TfidfModel.load(os.path.join(model_path, "tfidf.model"))
    p = sentence
    p_bow = dictionary.doc2bow((p.split()))
    p_tfidf = tfidf[p_bow]
    r_tfidf = []
    for i in p_tfidf:
        r_tfidf.append((new_dictionary.get(i[0]), i[1]))
    return r_tfidf


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    # 生成tf-idf模型
    # make_tf_idf_model(model_path="./tf_idf_models",
    #                   data_path="./train_data",
    #                   model_name="tfidf.model",
    #                   dictionary_name="tfidf.dictionary")
    # # 加载并使用模型
    ti = load_tf_idf_model("./tf_idf_models", "我 熟练 使用 电脑 玩 电子游戏 尤其 是 外星人 鸦片 罂粟")
    print(ti)
