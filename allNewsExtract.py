import os
import time
from pymongo import MongoClient
from TripleExtract.utlis import *
from TripleExtract.triple_extract import TripleExtractor
from TripleExtract.sentence_parser import SentenceParser

sent_parser = SentenceParser()  # 这里创建，多进程无法共享
# 注意 在 sentence_parser.py 中路径为 LTP_DIR = './ltp_data_v3.4.0'， 但是运行这个文件时，路径就不一样了./ 代表当前运行文件所在路径

from multiprocessing import Process
from multiprocessing import Pool, cpu_count





def extract_and_write(MongoURL, datebase, read_collection, write_collection, start, end):
    print("sub process, range:{}-{}, pid:{}".format(start, end, os.getpid()))
    Client = MongoClient(MongoURL)
    db = Client[datebase]
    read_collet = db[read_collection]
    write_collet = db[write_collection]


    data = read_collet.find({}).skip(start)
    for idx, item in enumerate(data):

        if idx+start > end : break

        result = {}
        result['_id'] = 'article-' + str(item['_id'])
        # print(item['title'])
        try:
            for sent in split_sentence(item['title']):
                if len(sent) > 6 and len(sent) < 100:
                    words, postags, nertags, child_dict_list, format_parse_list, roles_dict = sent_parser.parser_main(sent)
                    extractor = TripleExtractor(words, postags, nertags, child_dict_list, format_parse_list, roles_dict)
                    svos = extractor.extract()
                    # print(sent, svos)
                    result[sent.replace('.', '')] = svos


            for sent in split_sentence(item['content']):
                if len(sent) > 6 and len(sent) < 100:
                    words, postags, nertags, child_dict_list, format_parse_list, roles_dict = sent_parser.parser_main(sent)
                    extractor = TripleExtractor(words, postags, nertags, child_dict_list, format_parse_list, roles_dict)
                    svos = extractor.extract()
                    # print(sent, svos)
                    result[sent.replace('.', '')] = svos
        except Exception as e:
            print("extract error: ", e)
        # for i, sent in enumerate(title_sents):
        #     result[sent.replace('.', '')] = title_svos[i]
        # for i, sent in enumerate(content_sents):
        #     result[sent.replace('.', '')] = content_svos[i]

        try:
            write_collet.insert(result)
        except:
            print("write error, number:{}, id:{}".format(start+idx, result['_id']))






if __name__ == '__main__':

    MongoURL = "192.168.5.150:27017"
    datebase = 'Sina'
    read_collection = 'article20191121'
    write_collection = 'triple20191121'
    doc_count = MongoClient(MongoURL)[datebase][read_collection].count()

    cpu_num = cpu_count()
    cpu_num = 8   # 内存限制

    count = doc_count // cpu_num
    parts_range = [ [i*count,  i*count+count-1]  for i in range(cpu_num)]
    if(parts_range[-1][-1] < doc_count): parts_range[-1][-1] = doc_count

    start_time = time.time()
    p = Pool(cpu_num)
    rets = []
    for i in range(cpu_num):
        ret = p.apply_async(extract_and_write, args=(MongoURL, datebase, read_collection, write_collection, parts_range[i][0], parts_range[i][1]))
        rets.append(ret)

    for ret in rets:
        print(ret.get())  # 这样才能输出子程序异常

    p.close()
    p.join()

    end_time = time.time()

    print("总用时：{}".format(end_time-start_time))

    # 内存溢出是因为句子太长