import pkuseg
import numpy as np
from pyltp import SentenceSplitter
from Sentence import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer


seg = pkuseg.pkuseg(model_name='news')
seg_pos = pkuseg.pkuseg(model_name='news', postag=True)
vectorizer = TfidfVectorizer()



class TokenizeNews():
    """News that have been preprocessed

    这里的分词都没有remove stopwords

    Attributes:
        seg_pos_title: News的title的带词性分词数组
        sentences: News的content中包含的所有句子,为Sentence对象list
        descent_sentence_index: 综合计算之后得到的前N个句子，sentences的index
        topic_words:
        max_sentence_original_socre:
        token_weight_dic:


    """

    #     _id = ''
    def __init__(self):
        self.title = ''
        self.seg_pos_title = []
        self.sentences = []
        self.topic_words = []
        self.descend_sentence_index = []
        self.max_sentence_original_socre = 0
        self.token_weight_dic = {}
    #     seg_pos_content = []
    #     seg_pos_all_text = []
        self.event = None

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


def preprocessing(news):
    """
    对News进行预处理，对文本进行分词和词性标注
    这里的分词,没有去掉stopwords,也没有去掉标点符号

    Args:
        news_list: 原始的新闻list

    Return：
        预处理之后的新闻list
    """
    # tokenizeNews_list = []
    # for news in news_list:
    tokenizeNews = TokenizeNews()
    tokenizeNews.title = news.title
    tokenizeNews.seg_pos_title = seg_pos.cut(news.title)
    sentences = SentenceSplitter.split(news.content)  # Using ltp split sentence
    sentences = [x for x in sentences if x != '']
    for idx, x in enumerate(sentences):
        sentence = Sentence()
        sentence.text = x
        sentence.location = idx
        sentence.seg_pos = seg_pos.cut(x)
        tokenizeNews.sentences.append(sentence)
    # tokenizeNews_list.append(tokenizeNews)
    #
    # print(len(tokenizeNews_list))
    # return tokenizeNews_list
    return tokenizeNews

def tfidf(documents):
    """Calculate the tfidf value

    Args:
        documents: documents list and every document is a string connect with tokens
    """

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        vacabulary = vectorizer.get_feature_names()
        return tfidf_matrix, vacabulary
    except Exception as e:
        print("Error: {}".format(e))

def record_token_weight(tokenizeNews_list, all_tfidf_array, all_vocabulary):
    """
    对每个News中的记录所有的tokens和其对应的tiidf权重
    感觉这个可以用来作为TokenizeNews的成员函数，但是由于之后的操作都是基于这个词典的，如果想要执行后面的就必须执行这个
    这种在执行后面的时候，如果确保先执行了这个

    Args:
        tokenizeNews_list: 预处理以后的News list
        all_tfidf_array: 根据tokenizeNews list中的文本计算的tfidf矩阵

    """

    for row in range(all_tfidf_array.shape[0]):
        tokenizeNews_list[row].token_weight_dic = {}  # 为什么不加这句话会出错，会一直叠加，？？？
        for col in range(all_tfidf_array.shape[1]):
            if all_tfidf_array[row][col] > 0:
                tokenizeNews_list[row].token_weight_dic[all_vocabulary[col]] = all_tfidf_array[row][col]


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


# def sentence_sort(tokenizeNews_list):
def sentence_sort(tokenizeNews):
    """
    根据每个News的最终得分，进行一个降序排序，记录排序之后的index

    Args:
        这里参数的话是用list还是单个news，这要看你想怎么用了，之后再看怎么写比较好
        tokenizeNews_list:
        tokenizeNews:
    """
    score_list = [x.score for x in tokenizeNews.sentences]
    score_array = np.array(score_list)
    tokenizeNews.descend_sentence_index = np.argsort(-score_array)
    # for idx, tokenizeNews in enumerate(tokenizeNews_list):
    #     score_list = [x.score for x in tokenizeNews.sentences]
    #     score_array = np.array(score_list)
    #     tokenizeNews.descend_sentence_index = np.argsort(-score_array)


# def find_location(tokenizeNews_list, N=3):
def find_location(tokenizeNews, N=3):
    """找到News中的地理位置

    Version_1: 只是简单的通过POS的标注来判断是否是location
    Version_2: LSTM+CRF

    Problems:
        地点词'ns'，有些词不是地点词，也被标注成了地点词。（预处理一下会好吗，可以试试，去掉stopwords
            ('35个', 'ns')('11&', 'ns')('35&', 'ns')('74&', 'ns')('Sunwing）', 'ns')
            ('那', 'ns')('&&&埃塞俄比亚', 'ns')('[环', 'ns')(']肯尼亚', 'ns')
            还有一个问题，这里的location是确切的event发生的地点，还是和event相关的地点。（娱乐新闻应该和location没有太紧密的联系）再有就是看把event是用来干什么的，
            如果就是用来明确的表达一个event，那应该就是event发生的location，如果是用来clustering的，那就应该是与event相关的locations

    Args:
        tokenizeNews_list: 经过预处理之后的News列表
        N: 按照得分排序后的前N个句子

    Return:
        location_count字典

    """
    location_count_dic = {}

    # 从News的title中找location
    for item in tokenizeNews.seg_pos_title:
        if item[1] == 'ns':  # ns 地名
            if item[0] not in location_count_dic.keys():
                location_count_dic[item[0]] = 1
            else:
                location_count_dic[item[0]] += 1

    # 从News的前N个句子中找location
    for i, idx in enumerate(tokenizeNews.descend_sentence_index):
        if i < N:
            for item in tokenizeNews.sentences[idx].seg_pos:
                if item[1] == 'ns':
                    if item[0] not in location_count_dic.keys():
                        location_count_dic[item[0]] = 1
                    else:
                        location_count_dic[item[0]] += 1

    # print(location_count_dic)
    return location_count_dic
    # for idx, tokenizeNews in enumerate(tokenizeNews_list):
    #     location_count_dic = {}
    #
    #     # 从News的title中找location
    #     for item in tokenizeNews.seg_pos_title:
    #         if item[1] == 'ns':  # ns 地名
    #             if item[0] not in location_count_dic.keys():
    #                 location_count_dic[item[0]] = 1
    #             else:
    #                 location_count_dic[item[0]] += 1
    #
    #     # 从News的前N个句子中找location
    #     for i, idx in enumerate(tokenizeNews.descend_sentence_index):
    #         if i < N:
    #             for item in tokenizeNews.sentences[idx].seg_pos:
    #                 if item[1] == 'ns':
    #                     if item[0] not in location_count_dic.keys():
    #                         location_count_dic[item[0]] = 1
    #                     else:
    #                         location_count_dic[item[0]] += 1
    #
    #     print(location_count_dic)
#     return location_count_dic


# def find_date(tokenizeNews_list, N=3):
def find_date(tokenizeNews, N=3):
    """找到News发生的时间


    Version_1: 只是简单的通过POS的标注来判断是否是location
    Version_2: LSTM+CRF

    Problems:
        时间词't'
        由于通过POS识别出来的时间词有以下问题：
        1.不是绝对时间，更多的是相对时间，例如：”昨日“，”当天“等等
        2.时间残缺，例如缺少年份，月份，例如：“3月10日”，“10日”，“早晨八点”
        上面的两个问题，猜想可以根据新闻发布的时间进行补全或者推断，但是，前提是假定提出来的时间是和发布时间是有关联的。

        何老师：对于时间词的补全，可以在多事件上进行join操作，相互补全（相似度计算，相同的event进行一些确实维度的补全）

    Args:
        tokenizeNews_list: 经过预处理之后的News列表
        N: 按照得分排序后的前N个句子

    Return:
        date_count字典
    """

    date_count_dic = {}

    # 从News的title中找date
    for item in tokenizeNews.seg_pos_title:
        if item[1] == 't':
            if item[0] not in date_count_dic.keys():
                date_count_dic[item[0]] = 1
            else:
                date_count_dic[item[0]] += 1

    # 从News的前N个句子中找date
    for i, idx in enumerate(tokenizeNews.descend_sentence_index):
        if i < 5:
            for item in tokenizeNews.sentences[idx].seg_pos:
                if item[1] == 't':
                    if item[0] not in date_count_dic.keys():
                        date_count_dic[item[0]] = 1
                    else:
                        date_count_dic[item[0]] += 1

    # print(date_count_dic)
    return date_count_dic
    # for idx, tokenizeNews in enumerate(tokenizeNews_list):
    #     date_count_dic = {}
    #
    #     # 从News的title中找date
    #     for item in tokenizeNews.seg_pos_title:
    #         if item[1] == 't':
    #             if item[0] not in date_count_dic.keys():
    #                 date_count_dic[item[0]] = 1
    #             else:
    #                 date_count_dic[item[0]] += 1
    #
    #     # 从News的前N个句子中找date
    #     for i, idx in enumerate(tokenizeNews.descend_sentence_index):
    #         if i < 5:
    #             for item in tokenizeNews.sentences[idx].seg_pos:
    #                 if item[1] == 't':
    #                     if item[0] not in date_count_dic.keys():
    #                         date_count_dic[item[0]] = 1
    #                     else:
    #                         date_count_dic[item[0]] += 1
    #
    #     print(date_count_dic)
#     return date_count_dic
