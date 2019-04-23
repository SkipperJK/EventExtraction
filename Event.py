import os

'''
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
LTP_DATA_DIR = '/home/skipper/Downloads/ltp_data_v3.4.0'



# 没有加载模型成功，竟然不报错。。。
# 分词模型-load
seg_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
segmentor = Segmentor()
segmentor.load(seg_model_path)

# 词性标注模型-load
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger = Postagger()
postagger.load(pos_model_path)

# 依存句法分析模型-load
parser_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
parser = Parser()
parser.load(parser_model_path)

# 命名体识别模型-load
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)

# 语义角色标注模型-load
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')
labeller = SementicRoleLabeller()
labeller.load(srl_model_path)
'''


class Event():
    """Represent an event

    Attributes:
        data:
        locatioin:
        core_words:
        objects:

    """

    date = ""  # 暂定为news发表的日期
    location = ""
    core_words = []
    objects = []

    def show(self):
        print("Event detail:")
        print("\tWhen: %s" % (self.date))
        print("\tWhere: %s" % (self.location))
        print("\tOjects: %s" % (' '.join(self.objects)))
        print("\tCore Words: %s" % (' '.join(self.core_words)))

# define event class some function ,such as print???


def find_location(tokenizeNews_list, N=3):
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

    for idx, tokenizeNews in enumerate(tokenizeNews_list):
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

        print(location_count_dic)
#     return location_count_dic


def find_date(tokenizeNews_list, N=3):
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
    for idx, tokenizeNews in enumerate(tokenizeNews_list):
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

        print(date_count_dic)
#     return date_count_dic



'''
def srl(sentence):
    """Semantic Role Label

    感觉预处理的时候，最好去掉标点符号。

    Args:
        sentence: raw text of sentence

    Return:


    """

    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    arcs = parser.parse(words, postags)
    netags = recognizer.recognize(words, postags)

    roles = labeller.label(words, postags, arcs)
    who = None
    whom = None
    temporal = None
    location = None
    for role in roles:
        for arg in role.arguments:
            if arg.name == 'A0':
                who = get_object(words, arg.range.start, arg.range.end)
            #                 print("Who: %s"%who)
            if arg.name == 'A1':
                whom = get_object(words, arg.range.start, arg.range.end)
            #                 print("Whom: %s"%whom)
            if arg.name == 'TMP':
                temporal = get_object(words, arg.range.start, arg.range.end)
            #                 print("Temporal: %s"%temporal)
            if arg.name == 'LOC':
                location = get_object(words, arg.range.start, arg.range.end)
    #                 print("Location: %s"%location)

    segmentor.release()
    postagger.release()
    parser.release()
    recognizer.release()
    labeller.release()
    return who, whom, temporal, location


def get_object(words, start, end):
    """Transfer SRL to text according index

    Args:
        words: tokens list
        start: strat index of words list
        end: end index of words list
    """

    string = ''
    if start == end:
        string = words[start]
    else:
        for i in range(start, end + 1):
            string += words[i]
    return string


# sent = '国务院总理李克强调研上海外高桥时提出，支持上海积极探索新机制。'
# sent = '不到半年两起空难346人死亡 波音恐在劫难逃'
# sent = '全球最大航企美国航空:延长波音737MAX的停飞期'
# 国务院 (机构名) 总理李克强 (人名) 调研上海外高桥 (地名) 时提出，支持上海 (地名) 积极探索新机制。


def extract_arg(tokenizeNews_list):
    """
    对每个News提取subject和object
    这个函数要重新写，太冗余了

    Args：

    """

    for idx, tokenizeNews in enumerate(tokenizeNews_list):
        who = {}  # who_dict
        whom = {}
        temporal = {}
        location = {}
        w1 = []
        w2 = []
        t = []
        l = []
        #     #     print(idx)
        #         print(news_list[idx].title)
        temp1, temp2, temp3, temp4 = srl(tokenizeNews.title)
        #         print()
        w1.append(temp1)
        w2.append(temp2)
        t.append(temp3)
        l.append(temp4)
        for i, idx in enumerate(tokenizeNews.descend_sentence_index):
            if i < 3:
                #                 print(tokenizeNews.sentences[idx].text)
                temp1, temp2, temp3, temp4 = srl(tokenizeNews.sentences[idx].text)
                #                 print()
                w1.append(temp1)
                w2.append(temp2)
                t.append(temp3)
                l.append(temp4)

        for item in w1:
            if item in who.keys():
                who[item] += 1
            else:
                who[item] = 1

        for item in w2:
            if item in whom.keys():
                whom[item] += 1
            else:
                whom[item] = 1

        for item in t:
            if item in temporal.keys():
                temporal[item] += 1
            else:
                temporal[item] = 1

        for item in l:
            if item in location.keys():
                location[item] += 1
            else:
                location[item] = 1
        print("Who:", end=' ')
        print(who)
        print("Whom:", end=' ')
        print(whom)
        print("Temporal:", end=' ')
        print(temporal)
        print("Location:", end=' ')
        print(location)

#         break

'''

