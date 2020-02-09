import os.path


def check_model_exist(path):
    return os.path.exists(path)


if __name__ == "__main__":
    print(check_model_exist("./models/word2Vec.model"))
