import os


def compress_data(i):
    target = "D:\\workspace\\train_data\\data{}.7z".format(i)
    source_dir = "D:\\workspace\\python_src\\nlp_project\\train_data\\*"
    cmd = os.getcwd(
    ) + "\\7z\\7z.exe  a -t7z %s  %s* -r -mx=5 -m0=LZMA2 -ms=10m -mf=on -mhc=on -mmt=on" % (
        target, source_dir)
    print(cmd)
    os.system(cmd)
    print(source_dir, '=====>', target)


if __name__ == "__main__":
    compress_data(1)
