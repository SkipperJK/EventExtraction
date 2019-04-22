from pymongo import MongoClient
from News import News

MongoURL = "10.141.211.101:27017"
Client = MongoClient(MongoURL)
db = Client['admin']
db.authenticate('scidb', 'he')


def read_mongo(db, collection):
    """
    从MongoDB中读取数据

    Args:
        db: MongoDB数据库DB名
        collection: collection名
    Return:
        原始新闻list
    """
    db = Client[db]
    collection = db[collection]
    news_set = collection.find({})

    news_list = []
    for item in news_set:
        news = News()
        news.title = item['title']
        news.content = item['text']
        news.date = item['time']
        if "topic" in item.keys():
            news.topic = item['topic']
        for image in item['image']:
            news.images.append(image)
        news_list.append(news)
    print(len(news_list))
    return news_list




