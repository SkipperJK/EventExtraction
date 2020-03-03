import os
import config as config
import tensorflow as tf
from HotSpotServer.component.event_analyze.model import BiLSTM_CRF
from HotSpotServer.component.event_analyze.data import read_dictionary, tag2label, random_embedding
from HotSpotServer.component.event_analyze.utils import calculate_weight, count_entity
from django.test import TestCase

# get char embeddings
word2id = read_dictionary(config.WORD2ID_PATH)
embeddings = random_embedding(word2id, 300)
# paths setting
paths = {}
output_path = config.OUTPUT_PATH
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

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

ckpt_file = tf.train.latest_checkpoint(model_path)
paths['model_path'] = ckpt_file
model = BiLSTM_CRF(embeddings, tag2label, word2id, paths, config=tfconfig)
model.build_graph()
saver = tf.train.Saver()

# 加载训练好的模型参数
sess = tf.Session(config=tfconfig)
saver.restore(sess, ckpt_file)


def get_per_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-PER':
                if 'per' in locals().keys():
                    PER.append(per)
                    del per
                per = char
                if i+1 == length:
                    PER.append(per)
            if tag == 'I-PER':
                per += char
                if i+1 == length:
                    PER.append(per)
            if tag not in ['I-PER', 'B-PER']:
                if 'per' in locals().keys():
                    PER.append(per)
                    del per
                continue
    except:
        pass
    return PER


def get_loc_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-LOC':
                if 'loc' in locals().keys():
                    LOC.append(loc)
                    del loc
                loc = char
                if i+1 == length:
                    LOC.append(loc)
            if tag == 'I-LOC':
                loc += char
                if i+1 == length:
                    LOC.append(loc)
            if tag not in ['I-LOC', 'B-LOC']:
                if 'loc' in locals().keys():
                    LOC.append(loc)
                    del loc
                continue
    except:
        pass
    return LOC


def get_org_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-ORG':
                if 'org' in locals().keys():
                    ORG.append(org)
                    del org
                org = char
                if i+1 == length:
                    ORG.append(org)
            if tag == 'I-ORG':
                org += char
                if i+1 == length:
                    ORG.append(org)
            if tag not in ['I-ORG', 'B-ORG']:
                if 'org' in locals().keys():
                    ORG.append(org)
                    del org
                continue
    except:
        pass
    return ORG


def dnn_ner(sentence, ner_type):
    """
    对句子提取 指定类型 的实体
    :param sentence: 输入的句子
    :param ner_type: 实体的类型，取值为 per/loc/org，分别代表Person, Location, Organization
    :return:
    """
    demo_sent = sentence
    demo_sent = list(demo_sent.strip())
    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
    tag = model.demo_one(sess, demo_data)
    if ner_type == 'per':
        return get_per_entity(tag, demo_sent)
    if ner_type == 'loc':
        return get_loc_entity(tag, demo_sent)
    if ner_type == 'org':
        return get_org_entity(tag, demo_sent)


def dnn_ner_all(sentence):
    """
    对句子提取 所有类型 的实体
    :param sentence: 输入的句子
    :return: Person，Location，Organization三个实体列表
    """
    demo_sent = sentence
    demo_sent = list(demo_sent.strip())
    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
    tag = model.demo_one(sess, demo_data)
    per = get_per_entity(tag, demo_sent)
    loc = get_loc_entity(tag, demo_sent)
    org = get_org_entity(tag, demo_sent)
    return per, loc, org


def get_dnn_entity_count_dict(prep_article, entity_type, sentence_type='original', sentence_count=4):
    """
    对一篇文章的title和中心句进行进行 指定类型 实体提取（使用POS和LSTM+CRF两种方法结合），并计算权重
    :param prep_article: 预处理过的文章
    :param entity_type: 实体类型，取值为 date/per/loc/org 分别代表Date, Person, Location, Organization
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: 实体-权重字典
    """

    entities = []
    # entities.append([])
    # if entity_type == 'date':
    #     entity_type = 't'
    # if entity_type == 'per':
    #
    #     entity_type = 'nr'
    # if entity_type == 'loc':
    #     entity_type = 'ns'
    # if entity_type == 'org':
    #     entity_type = 'nt'


    # 文章title
    # print(prep_article.title)
    if entity_type == 'per':
        # entities.append(dnn_ner(prep_article.title, 'per'))
        pers = [per.replace(' ', '').replace('#', '') for per in dnn_ner(prep_article.title, 'per')]
        entities.append([per for per in pers if len(per) > 1])
    # 文章句子
    if sentence_count > 0:
        if entity_type == 'per':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    # print(sentence.text)
                    # entities.append(dnn_ner(sentence.text, 'per'))
                    pers = [per.replace(' ', '').replace('#', '') for per in dnn_ner(sentence.text, 'per')]
                    entities.append([per for per in pers if len(per) > 1])

    entity_count_dict = count_entity(entities)
    return entity_count_dict


def get_entity_weight_dict(prep_article, entity_type, sentence_type='original', sentence_count=4):
    """
    对一篇文章的title和中心句进行进行 指定类型 实体提取（使用POS和LSTM+CRF两种方法结合），并计算权重
    :param prep_article: 预处理过的文章
    :param entity_type: 实体类型，取值为 date/per/loc/org 分别代表Date, Person, Location, Organization
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: 实体-权重字典
    """
    if entity_type == 'date':
        entity_type = 't'
    if entity_type == 'per':
        entity_type = 'nr'
    if entity_type == 'loc':
        entity_type = 'ns'
    if entity_type == 'org':
        entity_type = 'nt'

    entities = []
    entities.append([])
    # 文章title
    for item in prep_article.seg_pos_title:
        if item[1] == entity_type:
            entities[0].append(item[0])
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    # entities.append([])
                    for item in sentence.seg_pos:
                        if item[1] == entity_type:
                            entities[i+1].append(item[0])
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    # entities.append([])
                    for item in prep_article.sentences[idx].seg_pos:
                        if item[1] == entity_type:
                            entities[i+1].append(item[0])

    entity_weight_dict = calculate_weight(entities)
    return entity_weight_dict


def get_ner_weight_dict(prep_article, sentence_type='original', sentence_count=4):
    """
    对一篇文章的title和中心句进行进行NER提取（使用POS和LSTM+CRF两种方法结合），并计算权重
    :param prep_article: 预处理过的文章
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: date, person, location, organization四个实体-权重字典
    """
    dates = []
    persons = []
    locs = []
    orgs = []
    # 文章title
    dates.append([])
    per_list, loc_list, org_list = dnn_ner_all(prep_article.title)
    persons.append(per_list)
    locs.append(loc_list)
    orgs.append(org_list)
    for item in prep_article.seg_pos_title:
        if item[1] == 't':
            dates[0].append(item[0])
        if item[1] == 'nr':
            persons[0].append(item[0])
        if item[1] == 'ns':
            locs[0].append(item[0])
        if item[1] == 'nt':
            orgs[0].append(item[0])
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    dates.append([])
                    per_list, loc_list, org_list = dnn_ner_all(sentence.text)
                    persons.append(per_list)
                    locs.append(loc_list)
                    orgs.append(org_list)
                    for item in sentence.seg_pos:
                        if item[1] == 't':
                            dates[i + 1].append(item[0])
                        if item[1] == 'nr':
                            persons[i + 1].append(item[0])
                        if item[1] == 'ns':
                            locs[i + 1].append(item[0])
                        if item[1] == 'nt':
                            orgs[i + 1].append(item[0])
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    dates.append([])
                    per_list, loc_list, org_list = dnn_ner_all(prep_article.sentences[idx].text)
                    persons.append(per_list)
                    locs.append(loc_list)
                    orgs.append(org_list)
                    for item in prep_article.sentences[idx].seg_pos:
                        if item[1] == 't':
                            dates[i + 1].append(item[0])
                        if item[1] == 'nr':
                            persons[i + 1].append(item[0])
                        if item[1] == 'ns':
                            locs[i + 1].append(item[0])
                        if item[1] == 'nt':
                            orgs[i + 1].append(item[0])

    date_weight_dict = calculate_weight(dates)
    person_weight_dict = calculate_weight(persons)
    location_weight_dict = calculate_weight(locs)
    org_weight_dict = calculate_weight(orgs)
    return date_weight_dict, person_weight_dict, location_weight_dict, org_weight_dict


class TestNER(TestCase):

    def test_ner(self):
        text = '71岁李保田三十多年不接广告，与妻子恩爱50多年，儿孙满堂'
        per = dnn_ner(text, 'per')
        print(per)
