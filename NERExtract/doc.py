"""
Basic data structure
"""
import numpy as np


class PreprocessArticle():
    """经过预处理的文章类

    将文章的title和content进行分词和词性标注
    其中content是Sentence的实例list

    Attributes:
        seg_pos_title: News的title的带词性分词数组
        sentences: News的content中包含的所有句子,为Sentence对象list
        descent_sentence_index: 综合计算之后得到的前N个句子，sentences的index
        topic_words:
        max_sentence_original_socre:
        token_weight_dict:
    """

    def __init__(self):
        self._id = ''
        self._date = ''
        self._title = ''
        self._seg_pos_title = []
        self._sentences = []
        self._descend_sentence_index = []
        self._max_sentence_original_socre = 0
        self._topic_words = []
        self._token_weight_dict = {}
        self._event = None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date):
        self._date = date

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def seg_pos_title(self):
        return self._seg_pos_title

    @seg_pos_title.setter
    def seg_pos_title(self, value):
        self._seg_pos_title = value

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sentences):
        self._sentences = sentences

    @property
    def descend_sentence_index(self):
        return self._descend_sentence_index

    @descend_sentence_index.setter
    def descend_sentence_index(self, descend_sentence_index):
        self._descend_sentence_index = descend_sentence_index

    @property
    def max_sentence_original_socre(self):
        return self._max_sentence_original_socre

    @max_sentence_original_socre.setter
    def max_sentence_original_socre(self, max_sentence_original_socre):
        self._max_sentence_original_socre = max_sentence_original_socre

    @property
    def topic_words(self):
        return self._topic_words

    @topic_words.setter
    def topic_words(self, topic_words):
        self._topic_words = topic_words

    @property
    def token_weight_dict(self):
        return self._token_weight_dict

    @token_weight_dict.setter
    def token_weight_dict(self, token_weight_dict):
        self._token_weight_dict = token_weight_dict

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, event):
        self._event = event

    def title_token_string(self):
        title = ''
        for x in self.seg_pos_title:
            title = title + ' ' + x[0]
        return title

    def content(self):
        content = ''
        for s in self.sentences:
            for x in s.seg_pos:
                content = content + ' ' + x[0]
        return content

    def all_text(self):
        return self.title_token_string() + ' ' + self.content()

    def sentents_count(self):
        return len(self.seg_pos_sentences)

    def calculate_sentence_original_score(self):
        for sentence in self._sentences:
            tokens = [x[0] for x in sentence.seg_pos]
            sentence_weight = sum(self._token_weight_dict[x] for x in tokens \
                                  if x in self._token_weight_dict.keys())
            if sentence_weight > self._max_sentence_original_socre:
                self._max_sentence_original_socre = sentence_weight
            sentence.score_original = sentence_weight

    def record_token_weight(self, row_number, tfidf_array, vocabulary):
        # print(row_number)
        for col in range(tfidf_array.shape[1]):
            if tfidf_array[row_number][col] > 0:
                self._token_weight_dict[vocabulary[col]] = tfidf_array[row_number][col]

    def extract_topicwords(self, row_number, tfidf_array, vocabulary, N=8):
        # np.argsort() asscend sort, 对每一行的tfidf进行一个降序排序，返回排序结果的索引矩阵indices
        descend_sort_index = np.argsort(-tfidf_array, axis=1)
        for idx in range(len(tfidf_array)):
            for i in range(N):
                point = descend_sort_index[row_number][i]
                if tfidf_array[row_number][point] > 0:
                    self._topic_words.append(vocabulary[point])

    def calculate_sentence_score(self):
        """
        计算每个句子的得分
        参数：
            1.句子的位置
            2.句子包含词的个数
            3.命名实体包含哪些
        最终的分数需要加权计算
        ***可以构建成一个函数
        """
        for sentence in self._sentences:

            # Term Weight Score
            score_term = sentence.score_original / self._max_sentence_original_socre

            # Sentence Location Score
            if sentence.location < 3:
                score_location = 1
            else:
                score_location = 0

            # Sentence Length Score
            if len(sentence.seg_pos) > 16:
                score_len = 1
            else:
                score_len = 0

            # Number of Name Nentities Score
            # Problem： 其实对于很多句子来说，很少有time，place, person, organization名词
            # 或者可能是这个pkuseg分词的问题，导致，socre_entity这个得分大部分为0.
            count = 0
            pos = ['t', 'nr', 'ns', 'nt', 'nz', 'j']
            for item in sentence.seg_pos:
                if item[1] in pos:
                    count = count + 1
            score_entity = count / len(sentence.seg_pos)

            # Title words overlap rate Score
            title_tokens = set([x[0] for x in self._seg_pos_title if x[0] in self._token_weight_dict.keys()])
            sentence_tokens = set([x[0] for x in sentence.seg_pos if x[0] in self._token_weight_dict.keys()])
            intersection_tokens = title_tokens & sentence_tokens
            title_weight = sum(self._token_weight_dict[x] for x in list(title_tokens))
            intersection_weight = sum(self._token_weight_dict[x] for x in list(intersection_tokens))
            if title_weight == 0:  # title_weight 有可能等于0
                score_title = 0
            else:
                score_title = intersection_weight / title_weight

            sentence.score_term = score_term
            sentence.score_location = score_location
            sentence.score_len = score_len
            sentence.score_entity = score_entity
            sentence.score_title = score_title

            sentence.score = 0.1 * score_term + 0.5 * score_location + 0.1 * score_len + \
                             0.1 * score_entity + 0.2 * score_title  # 加权计算(根据论文)

    def sentence_sort(self):
        score_list = [x.score for x in self._sentences]
        score_array = np.array(score_list)
        self._descend_sentence_index = np.argsort(-score_array)


class Sentence():
    """文章content字段中的每个句子类

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

    def __init__(self):
        self.text = ''
        self.seg_pos = []
        self.location = 0
        self.score_original = 0

        self.score_term = 0
        self.score_location = 0
        self.score_len = 0
        self.score_entitiy = 0
        self.score_title = 0
        self.score = 0

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


class Event():
    """事件类

    Attributes:
        data:
        locatioin:
        core_words:
        objects:

    """

    def __init__(self):
        self.release_date = ""
        self.date = ""  # 暂定为news发表的日期
        self.location = ""
        # core_words = []
        # objects = []
        self.who_count_dict = {}
        self.whom_count_dict = {}
        self.predicate_count_dict = {}
        self.loc_count_dict = {}
        self.date_count_dict = {}
        self.per_count_dict = {}
        self.loc_count_dict_dnn = {}
        self.org_count_dict = {}
        self.phrases = []

    def get_phrases(self):
        verbose = ''
        for phrase in self.phrases:
            verbose = verbose + '\t\t' + phrase.get() + '\n'
        return verbose

    def show(self):
        print("Event detail:")
        print("\tWhen: %s" % (self.date))
        print("\tWhere: %s" % (self.location))
        # print("\tOjects: %s" % (' '.join(self.objects)))
        # print("\tCore Words: %s" % (' '.join(self.core_words)))
        print("\twho_count_dict:", self.who_count_dict)
        print("\twhom_count_dict:", self.whom_count_dict)
        print("\tpredicate_count_dict:", self.predicate_count_dict)
        print("\tloc_count_dict:", self.loc_count_dict)
        print("\tdate_count_dict:", self.date_count_dict)
        print("\tper_count_dict:", self.per_count_dict)
        print("\tloc_count_dict_dnn:", self.loc_count_dict_dnn)
        print("\torg_count_dict:", self.org_count_dict)


class Phrase():

    def __init__(self):
        self.subject = []
        self.object = []
        self.predicate = ''

    def get_verbs(self):    # 有可能宾语是动词
        verbs = [self.predicate]
        for o in self.object:
            for word in o.split(' '):
                verbs.append(word)
        if len(verbs) > 1:
            verbs.pop()
        return verbs

    def get(self):
        # phrase = ','.join(self.subject) + " " + self.predicate + " " + ','.join(self.object)
        phrase = []
        phrase.append(''.join(self.subject))
        phrase.append(self.predicate)
        phrase.append(''.join(self.object))
        return phrase

    def show(self):
        phrase = ','.join(self.subject) + " " + self.predicate + " " + ','.join(self.object)
        print(phrase)
