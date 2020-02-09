import os
import shutil

shutil.rmtree("./train_data")  # 能删除该文件夹和文件夹下所有文件
os.mkdir("./train_data")
