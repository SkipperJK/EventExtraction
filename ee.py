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
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
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
output_path = os.path.join('.', train_data+"_save", timestamp)
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

def execute(sentence):

    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)

        demo_sent = sentence
        demo_sent = list(demo_sent.strip())
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        PER, LOC, ORG = get_entity(tag, demo_sent)
        print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))



news_list = read_mongo("Sina", "737news")
tokenizeNews_list = preprocessing(news_list)

all_texts = [tokenizeNews.all_text() for tokenizeNews in tokenizeNews_list]
all_tfidf_matrix, all_vocabulary = tfidf(all_texts)
all_tfidf_array = all_tfidf_matrix.toarray() # transfer scipy.crs.crs_matirx to numpy narray


record_token_weight(tokenizeNews_list, all_tfidf_array, all_vocabulary)

extract_news_topicwords(tokenizeNews_list, all_tfidf_array, all_vocabulary, N = 8)

calculate_sentence_original_score(tokenizeNews_list)

calculate_sentence_score(tokenizeNews_list)

sentence_sort(tokenizeNews_list)

find_location(tokenizeNews_list, N = 5)

find_date(tokenizeNews_list, N = 5)

for tokenizeNews in tokenizeNews_list:
    execute(tokenizeNews.title)
    for i, idx in enumerate(tokenizeNews.descend_sentence_index):
        if i < 4:
            execute(tokenizeNews.sentences[idx].text)
