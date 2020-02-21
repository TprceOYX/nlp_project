from opencc import OpenCC
cc1 = OpenCC("s2t")
text1 = "凉宫春日的忧郁"
convert1 = cc1.convert(text1)
print(convert1)
cc2 = OpenCC("t2s")
text2 = "涼宮春日的憂鬱"
convert2 = cc2.convert(text2)
print(convert2)
