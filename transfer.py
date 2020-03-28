from pymongo import MongoClient

MongoURLRead = '127.0.0.1:27017'
# MongoURL = "10.141.212.160:27017"
MongoURLWrite = "192.168.5.150:27017"
client_r = MongoClient(MongoURLRead)
db_r = client_r['Sina']
# db.authenticate('scidb', 'he')
collection_r = db_r['article20191121']

client_w = MongoClient(MongoURLWrite)
db_w = client_w['Sina']
collection_w = db_w['article20191121']



for item in collection_r.find({}):
    if not collection_w.find_one({'_id': item['_id']}):
        collection_w.insert(item)
