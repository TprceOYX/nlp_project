import re


pattern = re.compile(u'[\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]+')
str1 = "まみむめも"
m = pattern.match(str1)
if m is None:
    print("fail")
else:
    print(m.group())
