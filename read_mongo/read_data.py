import pymongo
# 连接数据库
myclient = pymongo.MongoClient(
    "mongodb://tprce:1634834938@47.94.0.240:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass%20Community&ssl=false"
)
mydb = myclient["document"]
mycol = mydb["baike"]
myresult = mycol.find({}, {"_id": 0, "text": 1}).limit(1)
for x in myresult:
    print(type(x["text"]))
