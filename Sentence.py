import numpy as np




class Sentence():
    """Every sentence of News content

    Attributes:
        text: raw sentence
        seg_pos:
        location:
        score_original:
        score_term:
        socre_location:
        score_len:
        score_entity:
        score_title:
        score: finally score
    """

    text = ''
    seg_pos = []
    location = 0
    score_original = 0

    score_term = 0
    score_location = 0
    score_len = 0
    score_entitiy = 0
    score_title = 0
    score = 0

    def show(self):
        print(self.text)
        print("\t", self.seg_pos)
        print("\tLocation: %d, score_original: %f" % (self.location, self.score_original))
        print("\tterm: %f, location: %d, len: %d, entity: %f, title: %f" % ( \
            self.score_term, self.score_location, \
            self.score_len, self.score_entitiy, \
            self.score_title))
        print("\tScore: %f" % self.score)

    def tokens(self):
        return [x[0] for x in self.seg_pos]








def extract_news_topicwords(tokenizeNews_list, all_tfidf_array, all_vocabulary, N=8):
    """
    每篇News提取根据ifidf权重对应的TopN个词

    Args:
        tokenizeNews_list: 预处理以后的News list
        all_tfidf_array: 根据tokenizeNews list中的文本计算的tfidf矩阵
        N： 前N个tfidf值对应的word


    """

    # np.argsort() asscend sort, 对每一行的tfidf进行一个降序排序，返回排序结果的索引矩阵indices
    descend_sort_index = np.argsort(-all_tfidf_array, axis=1)

    for idx in range(len(all_tfidf_array)):
        topic_words = []
        for i in range(N):
            point = descend_sort_index[idx][i]
            if all_tfidf_array[idx][point] > 0:
                topic_words.append(all_vocabulary[point])
        tokenizeNews_list[idx].topic_words = topic_words
    #     break
    # print(topic_words)


def calculate_sentence_original_score(tokenizeNews_list):

    """
    根据tfidf权重，对一个News计算每个句子的权重,并记录权重最高的句子的权重

    Args:
        tokenizeNews_list: 预处理以后的News list
    """

    for idx, tokenizeNews in enumerate(tokenizeNews_list):
        max_sentence_original_socre = 0
        for s in tokenizeNews.sentences:
            tokens = [x[0] for x in s.seg_pos]
            sentence_weight = sum(tokenizeNews.token_weight_dic[x] for x in tokens \
                                  if x in tokenizeNews.token_weight_dic.keys())
            if sentence_weight > max_sentence_original_socre:
                max_sentence_original_socre = sentence_weight
            s.score_original = sentence_weight
        tokenizeNews.max_sentence_original_socre = max_sentence_original_socre
    #     print(tokenizeNews.max_sentence_original_socre)
    #     break


# 其实这个方法，应该放在Topic类中，作为成员函数，使用的话应该topic.calculate_sentence_score(tokenizeNews_list)
def calculate_sentence_score(tokenizeNews_list):
    """
    计算每个句子的得分
    参数：
        1.句子的位置
        2.句子包含词的个数
        3.命名实体包含哪些

    Args:
        这里参数的话是用list还是单个news，这要看你想怎么用了，之后再看怎么写比较好
        tokenizeNews_list:
        tokenizeNews:

    最终的分数需要加权计算
    ***可以构建成一个函数
    """
    for idx, tokenizeNews in enumerate(tokenizeNews_list):
        for s in tokenizeNews.sentences:

            # Term Weight
            score_term = s.score_original / tokenizeNews.max_sentence_original_socre

            # Sentence Location
            if s.location < 3:
                score_location = 1
            else:
                score_location = 0

            # Sentence Length
            if len(s.seg_pos) > 16:
                score_len = 1
            else:
                score_len = 0

            # Number of Name Nentities
            # Problem： 其实对于很多句子来说，很少有time，place, person, organization名词
            # 或者可能是这个pkuseg分词的问题，导致，socre_entity这个得分大部分为0.
            count = 0
            pos = ['t', 'nr', 'ns', 'nt', 'nz', 'j']
            for item in s.seg_pos:
                if item[1] in pos:
                    count = count + 1
            score_entity = count / len(s.seg_pos)

            # Title words overlap rate
            title_tokens = set([x[0] for x in tokenizeNews.seg_pos_title \
                                if x[0] in tokenizeNews.token_weight_dic.keys()])
            sentence_tokens = set([x[0] for x in s.seg_pos \
                                   if x[0] in tokenizeNews.token_weight_dic.keys()])
            intersection_tokens = title_tokens & sentence_tokens
            title_weight = sum(tokenizeNews.token_weight_dic[x] for x in list(title_tokens))
            intersection_weight = sum(tokenizeNews.token_weight_dic[x] for x in list(intersection_tokens))
            if title_weight == 0:  # title_weight 有可能等于0
                score_title = 0
            else:
                score_title = intersection_weight / title_weight

            s.score_term = score_term
            s.score_location = score_location
            s.score_len = score_len
            s.score_entity = score_entity
            s.score_title = score_title

            s.score = 0.1 * score_term + 0.5 * score_location + 0.1 * score_len + \
                      0.1 * score_entity + 0.2 * score_title  # 加权计算(根据论文)
    #     break


def sentence_sort(tokenizeNews_list):
    """
    根据每个News的最终得分，进行一个降序排序，记录排序之后的index

    Args:
        这里参数的话是用list还是单个news，这要看你想怎么用了，之后再看怎么写比较好
        tokenizeNews_list:
        tokenizeNews:
    """
    for idx, tokenizeNews in enumerate(tokenizeNews_list):
        score_list = [x.score for x in tokenizeNews.sentences]
        score_array = np.array(score_list)
        tokenizeNews.descend_sentence_index = np.argsort(-score_array)