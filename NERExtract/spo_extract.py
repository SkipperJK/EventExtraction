import os
import config as config
from pyhanlp import HanLP
from pyltp import Segmentor, Postagger, Parser
from NERExtract.utils import calculate_weight
# 分词模型-load
seg_model_path = os.path.join(config.LTP_DATA_DIR, 'cws.model')
segmentor = Segmentor()
segmentor.load(seg_model_path)

# 词性标注模型-load
pos_model_path = os.path.join(config.LTP_DATA_DIR, 'pos.model')
postagger = Postagger()
postagger.load(pos_model_path)

# 依存句法分析模型-load
parser_model_path = os.path.join(config.LTP_DATA_DIR, 'parser.model')
ltp_parser = Parser()
ltp_parser.load(parser_model_path)




def get_ltp_sub_entity(word_list, arcs):
    subs = []
    for i, arc in enumerate(arcs):
        if arc.relation == "SBV":  # subject verb
            subs.append(word_list[i])
    return subs


def get_ltp_obj_entity(word_list, arcs):
    objs = []
    for i, arc in enumerate(arcs):
        if arc.relation == "VOB":  # object verb
            objs.append(word_list[i])
    return objs


def get_ltp_predicate_entity(word_list, arcs):
    predicates = []
    for i, arc in enumerate(arcs):
        if arc.relation == "HED":
            predicates.append(word_list[i])
    return predicates


# def get_ltp_entity_weight_dict(preprocess_article, entity_type, n=4):
def get_ltp_entity_weight_dict(prep_article, entity_type, sentence_type='original', sentence_count=4):
    """
    对每篇文章title和中心句子提取 subject/object/predicate, 并对对应的类型计算每个词的权重
    :param prep_article: PreprocessArticle类的实例
    :param entity_type: 提取词的类型，取值为 sub/obj/predicate, 分别代表 subject, object, predicate
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: subject/object/predicate的词-权重字典
    """
    entitis = []
    # 文章title
    words = segmentor.segment(prep_article.title)
    word_list = list(words)
    postags = postagger.postag(words)
    arcs = ltp_parser.parse(words, postags)
    if entity_type == 'sub':
        entitis.append(get_ltp_sub_entity(word_list, arcs))
    if entity_type == 'obj':
        entitis.append(get_ltp_obj_entity(word_list, arcs))
    if entity_type == 'predicate':
        entitis.append(get_ltp_predicate_entity(word_list, arcs))

    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    words = segmentor.segment(sentence.text)
                    word_list = list(words)
                    postags = postagger.postag(words)
                    arcs = ltp_parser.parse(words, postags)
                    if entity_type == 'sub':
                        entitis.append(get_ltp_sub_entity(word_list, arcs))
                    if entity_type == 'obj':
                        entitis.append(get_ltp_obj_entity(word_list, arcs))
                    if entity_type == 'predicate':
                        entitis.append(get_ltp_predicate_entity(word_list, arcs))
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    words = segmentor.segment(prep_article.sentences[idx].text)
                    word_list = list(words)
                    postags = postagger.postag(words)
                    arcs = ltp_parser.parse(words, postags)
                    if entity_type == 'sub':
                        entitis.append(get_ltp_sub_entity(word_list, arcs))
                    if entity_type == 'obj':
                        entitis.append(get_ltp_obj_entity(word_list, arcs))
                    if entity_type == 'predicate':
                        entitis.append(get_ltp_predicate_entity(word_list, arcs))
    entity_weight_dict = calculate_weight(entitis)
    return entity_weight_dict


def get_ltp_spo_weight_dict(prep_article, sentence_type='original', sentence_count=4):
    """
    对每篇文章title和中心句子提取subject, object, predicate, 并对subject, object, predicate计算每个词的权重
    :param prep_article: PreprocessArticle类的实例
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: subject, object, predicate的词-权重字典
    """
    subs = []
    objs = []
    predicates = []
    # 文章title
    words = segmentor.segment(prep_article.title)
    word_list = list(words)
    postags = postagger.postag(words)
    arcs = ltp_parser.parse(words, postags)
    subs.append(get_ltp_sub_entity(word_list, arcs))
    objs.append(get_ltp_obj_entity(word_list, arcs))
    predicates.append(get_ltp_predicate_entity(word_list, arcs))
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    words = segmentor.segment(sentence.text)
                    word_list = list(words)
                    postags = postagger.postag(words)
                    arcs = ltp_parser.parse(words, postags)
                    subs.append(get_ltp_sub_entity(word_list, arcs))
                    objs.append(get_ltp_obj_entity(word_list, arcs))
                    predicates.append(get_ltp_predicate_entity(word_list, arcs))
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    words = segmentor.segment(prep_article.sentences[idx].text)
                    word_list = list(words)
                    postags = postagger.postag(words)
                    arcs = ltp_parser.parse(words, postags)
                    subs.append(get_ltp_sub_entity(word_list, arcs))
                    objs.append(get_ltp_obj_entity(word_list, arcs))
                    predicates.append(get_ltp_predicate_entity(word_list, arcs))

    sub_weight_dict = calculate_weight(subs)
    obj_weight_dict = calculate_weight(objs)
    predicate_weight_dict = calculate_weight(predicates)
    return sub_weight_dict, obj_weight_dict, predicate_weight_dict


def get_hanlp_sub_entity(words):
    subs = []
    for word in words:
        if word.DEPREL == '主谓关系':
            subs.append(word.LEMMA)
    return subs


def get_hanlp_obj_entity(words):
    objs = []
    for word in words:
        if word.DEPREL == '动宾关系':
            objs.append(word.LEMMA)
    return objs


def get_hanlp_predicate_entity(words):
    predicates =[]
    for word in words:
        if word.DEPREL == '核心关系':
            predicates.append(word.LEMMA)
    return predicates


# def get_phrase_detail(words):
#     phrases = []
#     for word in words:
#         if word.DEPREL == '核心关系' and word.CPOSTAG == 'v':
#             # phrases.append(extract_detail(word, words))
#             phrases = phrases + extract_event_detail(word, words)
#     return phrases


# def get_hanlp_entity_weight_dict(preprocess_article, entity_type, n=4):
def get_hanlp_entity_weight_dict(prep_article, entity_type, sentence_type='original', sentence_count=4):

    """
    对每篇文章title和中心句子提取 subject/object/predicate, 并对对应的类型计算每个词的权重
    :param prep_article: PreprocessArticle类的实例
    :param entity_type: 提取词的类型，取值为 sub/obj/predicate, 分别代表 subject, object, predicate
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: subject/object/predicate的词-权重字典
    """
    entitis = []
    # 文章title
    words = HanLP.parseDependency(prep_article.title).word
    if entity_type == 'sub':
        entitis.append(get_hanlp_sub_entity(words))
    if entity_type == 'obj':
        entitis.append(get_hanlp_obj_entity(words))
    if entity_type == 'predicate':
        entitis.append(get_hanlp_predicate_entity(words))
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    words = HanLP.parseDependency(sentence.text).word
                    if entity_type == 'sub':
                        entitis.append(get_hanlp_sub_entity(words))
                    if entity_type == 'obj':
                        entitis.append(get_hanlp_obj_entity(words))
                    if entity_type == 'predicate':
                        entitis.append(get_hanlp_predicate_entity(words))
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    words = HanLP.parseDependency(prep_article.sentences[idx].text).word
                    if entity_type == 'sub':
                        entitis.append(get_hanlp_sub_entity(words))
                    if entity_type == 'obj':
                        entitis.append(get_hanlp_obj_entity(words))
                    if entity_type == 'predicate':
                        entitis.append(get_hanlp_predicate_entity(words))

    entity_weight_dict = calculate_weight(entitis)
    return entity_weight_dict


# def get_hanlp_spo_weight_dict(preprocess_article, n=4):
def get_hanlp_spo_weight_dict(prep_article, sentence_type='original', sentence_count=4):
    """
    对每篇文章title和中心句子提取subject, object, predicate 对subject, object, predicate计算每个词的权重
    :param prep_article: PreprocessArticle类的实例
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: subject, object, predicate的词-权重字典，phrase列表
    """
    subs = []
    objs = []
    predicates = []
    # phrases = []
    # 文章title
    words = HanLP.parseDependency(prep_article.title).word
    subs.append(get_hanlp_sub_entity(words))
    objs.append(get_hanlp_obj_entity(words))
    predicates.append(get_hanlp_predicate_entity(words))
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    words = HanLP.parseDependency(sentence.text).word
                    subs.append(get_hanlp_sub_entity(words))
                    objs.append(get_hanlp_obj_entity(words))
                    predicates.append(get_hanlp_predicate_entity(words))
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    words = HanLP.parseDependency(prep_article.sentences[idx].text).word
                    subs.append(get_hanlp_sub_entity(words))
                    objs.append(get_hanlp_obj_entity(words))
                    predicates.append(get_hanlp_predicate_entity(words))

    sub_weight_dict = calculate_weight(subs)
    obj_weight_dict = calculate_weight(objs)
    predicate_weight_dict = calculate_weight(predicates)
    return sub_weight_dict, obj_weight_dict, predicate_weight_dict


# def dnn_parse(sentence):
#     r1 = HanLP.parseDependency("刚刚，埃塞俄比亚航空集团CEO抵达飞机坠毁现场，确认无乘客生还，从埃航推特上发布的照片看，航班残片遍地都是。")
#     print(r1)
#     for word in r1.word:
#         #     print(word.ID, word.LEMMA, word.CPOSTAG, word.HEAD.ID, word.DEPREL)
#         if word.DEPREL == '核心关系':
#             s_p(r1.word, word)
