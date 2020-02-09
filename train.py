import logging
import multiprocessing
import os.path
import sys

from gensim.models import word2vec


def train_wordVectors(
    save_path,
    sentences,
    embedding_size=128,
    window=5,
    min_count=5,
):
    '''
    :param sentences: sentences可以是LineSentence或者PathLineSentences读取的文件对象，也可以是
                    The `sentences` iterable can be simply a list of lists of tokens,如lists=[['我','是','中国','人'],['我','的','家乡','在','广东']]
    :param embedding_size: 词嵌入大小
    :param window: 窗口
    :param min_count:Ignores all words with total frequency lower than this.
    :return: w2vModel
    '''
    if check_model_exist(save_path):
        w2vModel = word2vec.Word2Vec.load(save_path)
        w2vModel.build_vocab(sentences, update=True)
    else:
        w2vModel = word2vec.Word2Vec(sentences,
                                     size=embedding_size,
                                     window=window,
                                     min_count=min_count,
                                     workers=multiprocessing.cpu_count())
    w2vModel.train(sentences,
                   total_examples=w2vModel.corpus_count,
                   epochs=w2vModel.iter)
    return w2vModel


def save_wordVectors(w2vModel, word2vec_path):
    w2vModel.save(word2vec_path)
    # w2vModel.wv.save_word2vec_format("./models/word2Vec2.model", binary=False)


def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel


def check_model_exist(path):
    return os.path.exists(path)


def main_func():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    # 1.format: 指定输出的格式和内容，format可以输出很多有用信息，
    # %(asctime)s: 打印日志的时间
    # %(levelname)s: 打印日志级别名称
    # %(message)s: 打印日志信息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    # 打印这是一个通知日志
    logger.info("running %s" % ' '.join(sys.argv))
    # [1]若只有一个文件，使用LineSentence读取文件
    # segment_path='./data/segment/segment_0.txt'
    # sentences = word2vec.LineSentence(segment_path)

    # [1]若存在多文件，使用PathLineSentences读取文件列表

    segment_dir = './train_data'
    sentences = word2vec.PathLineSentences(segment_dir)
    # 一般训练，设置以下几个参数即可：
    word2vec_path = './models/word2Vec.model'
    model = train_wordVectors(word2vec_path,
                              sentences,
                              embedding_size=128,
                              window=5,
                              min_count=5)
    save_wordVectors(model, word2vec_path)
    # print(model2.wv.similarity('沙瑞金', '高育良'))


if __name__ == '__main__':
    main_func()
