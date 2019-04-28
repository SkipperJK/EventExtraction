import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding

from News import *
from preprocessing import *
from TokenizeNews import *
from Sentence import *
from Event import *

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()

## get char embeddings
word2id = read_dictionary(os.path.join('.', 'data_path', 'word2id.pkl'))
embeddings = random_embedding(word2id, 300)

train_data = 'data_path'
## paths setting
paths = {}
timestamp = str(int(time.time()))
output_path = os.path.join('.', train_data + "_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path

model_path = './data_path_save/1521112368/checkpoints'
ckpt_file = tf.train.latest_checkpoint(model_path)
print(ckpt_file)
paths['model_path'] = ckpt_file
model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
model.build_graph()
saver = tf.train.Saver()

# 加载训练好的模型参数
# with tf.Session(config=config) as sess:
sess = tf.Session(config=config)
saver.restore(sess, ckpt_file)


def execute(sentence):


    demo_sent = sentence
    demo_sent = list(demo_sent.strip())
    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
    tag = model.demo_one(sess, demo_data)
    PER, LOC, ORG = get_entity(tag, demo_sent)
    # print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
    return PER, LOC, ORG

import time
fp = open("./log.txt", "a")
fp.write("Begin:%s \n"%time.asctime(time.localtime()))

news_list = read_mongo("Sina", "737news")
tokenizeNews_list = []

for news in news_list:
    tokenizeNews_list.append(preprocessing(news))

all_texts = [tokenizeNews.all_text() for tokenizeNews in tokenizeNews_list]
all_tfidf_matrix, all_vocabulary = tfidf(all_texts)
all_tfidf_array = all_tfidf_matrix.toarray()  # transfer scipy.crs.crs_matirx to numpy narray

record_token_weight(tokenizeNews_list, all_tfidf_array, all_vocabulary)
extract_news_topicwords(tokenizeNews_list, all_tfidf_array, all_vocabulary, N=8)

# 计算每个句子的TfIdf分数
for tokenizeNews in tokenizeNews_list:
    for sentence in tokenizeNews.sentences:
        calculate_sentence_original_score(tokenizeNews, sentence)

# 计算每个句子加权分数
for tokenizeNews in tokenizeNews_list:
    for sentence in tokenizeNews.sentences:
        calculate_sentence_score(tokenizeNews, sentence)

# 对每个News中的句子进行排序
for tokenizeNews in tokenizeNews_list:
    sentence_sort(tokenizeNews)

# 根据POS记录每个事件的时间词和地点词
event_list = []
for tokenizeNews in tokenizeNews_list:
    event = Event()
    event.date_count_dic = find_date(tokenizeNews, N=5)
    event.loc_count_dic = find_location(tokenizeNews, N=5)
    event_list.append(event)

for j, tokenizeNews in enumerate(tokenizeNews_list):
    fp.write("第 %d 个新闻：\n"%j)

    PER = []
    LOC = []
    ORG = []
    tmp_per = []
    tmp_loc = []
    tmp_org = []
    tmp_per, tmp_loc, tmp_org = execute(tokenizeNews.title)
    fp.write("\t%s\n"%tokenizeNews.title)
    fp.write("\t\t {} {} {}\n".format(tmp_per, tmp_loc, tmp_org))
    PER += tmp_per
    LOC += tmp_loc
    ORG += tmp_org
    for i, idx in enumerate(tokenizeNews.descend_sentence_index):
        if i < 4:
            tmp_per = []
            tmp_loc = []
            tmp_org = []
            tmp_per, tmp_loc, tmp_org = execute(tokenizeNews.sentences[idx].text)
            fp.write("\t%s\n"%tokenizeNews.sentences[idx].text)
            fp.write("\t\t {} {} {}\n".format(tmp_per, tmp_loc, tmp_org))
            PER += tmp_per
            LOC += tmp_loc
            ORG += tmp_org
        else:
            break
    per_count_dic = {}
    org_count_dic = {}
    loc_count_dic_dnn = {}
    for person in PER:
        if person not in  per_count_dic.keys():
            per_count_dic[person] = 1
        else:
            per_count_dic[person] += 1
    for location in LOC:
        if location not in loc_count_dic_dnn.keys():
            loc_count_dic_dnn[location] = 1
        else:
            loc_count_dic_dnn[location] += 1
    for organization in ORG:
        if organization not in org_count_dic.keys():
            org_count_dic[organization] = 1
        else:
            org_count_dic[organization] += 1

    # 先假设新闻事件为发布时间
    event_list[j].date = news_list[j].date
    event_list[j].per_count_dic = per_count_dic
    event_list[j].loc_count_dic_dnn = loc_count_dic_dnn
    event_list[j].org_count_dic = org_count_dic
    fp.write("第 %d 个事件：\n"%j)
    fp.write("Event detail: \n")
    fp.write("\tWhen: %s\n" % (event_list[j].date))
    fp.write("\tWhere: %s\n" % (event_list[j].location))
    fp.write("\tloc_count_dic: {}\n".format(event_list[j].loc_count_dic))
    fp.write("\tdate_count_dic: {}\n".format(event_list[j].date_count_dic))
    fp.write("\tper_count_dic: {}\n".format(event_list[j].per_count_dic))
    fp.write("\tloc_count_dic_dnn: {}\n".format(event_list[j].loc_count_dic_dnn))
    fp.write("\torg_count_dic: {}\n".format(event_list[j].org_count_dic))

fp.close()
for event in event_list:
    event.show()
