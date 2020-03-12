import codecs

file = codecs.open("a.txt", "r", encoding="utf8")
data = []
for line in file:
    data.append(line.strip())
print("".join(data))
