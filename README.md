# nlp_project

# 使用说明

在命令行模式进入项目所在目录，然后输入python main.py 文件1路径 文件2路径

文件1路径和文件2路径为要查重的两个文本

# 运行环境

Python3.6.5

第三方包：gensim,opencc,jieba

# 文件说明

## data文件夹

dict.txt.big：jieba分词时用到的词典

stop_words.txt：jieba分词时用到的停用词

## models文件夹

final中保存的是二进制格式的word2vec，不可继续进行增量训练

## read_mongo文件夹

read_data.py：处理原始语料（爬取到的文本和维基百科词条）

## test文件夹

测试一些小功能

## tf_idf_models

保存tf-idf模型和它的词典

## train_data

用于制作模型的训练语料

## main.py

主函数，用于计算文本相似度并判断是否重复

## process_train_data.py

调用read_data.py中的各种语料的处理方法，将原始语料制作成训练语料保存到train_data中

## train_tf_idf.py

训练TF-IDF模型

## train_w2v.py

训练word2vec模型

