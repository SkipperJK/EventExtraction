import logging, sys, argparse
import pkuseg
from pyltp import SentenceSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from HotSpotServer.component.event_analyze.doc import PreprocessArticle
from HotSpotServer.component.event_analyze.doc import Sentence
from HotSpotServer.utils.stopwords import stopwords
seg_pos = pkuseg.pkuseg(model_name='news', postag=True)  # 加载模型，千万不要放在函数里面，要不每次执行一次又要加载一次模型，耗时


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# def s_p(word, words):
#     word_id = word.ID
#     predicate = word.LEMMA
#     who_list = []
#     whom_list = []
#     for word in words:
#         if word.HEAD.ID == word_id:
#             if word.DEPREL == '主谓关系':
#                 who_list.append(word.LEMMA)
#             if word.DEPREL == '动宾关系':
#                 whom_list.append(word.LEMMA)
#             if word.DEPREL == '并列关系':
#                 s_p(words, word)
#     for who in who_list:
#         print(who, end="、")
#     print(predicate, end=' ')
#     for whom in whom_list:
#         print(whom, end='、')
#     print()


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def article_preprocessing(article):
    """
    对News进行预处理，对文本进行分词和词性标注
    这里的分词,没有去掉stopwords,也没有去掉标点符号

    Args:
        Article: 原始的新闻Article

    Return：
        预处理之后的新闻PreprocessArticle
    """
    # print("article preprocessing")
    pre_article = PreprocessArticle()
    # pre_article.id = article.id
    pre_article.date = article.time
    pre_article.title = article.title
    pre_article.seg_pos_title = seg_pos.cut(article.title)
    sentences = SentenceSplitter.split(article.content)  # Using ltp split sentence
    sentences = [x for x in sentences if x != '']
    for idx, x in enumerate(sentences):
        sentence = Sentence()
        sentence.text = x
        sentence.location = idx
        sentence.seg_pos = seg_pos.cut(x)
        pre_article.sentences.append(sentence)
    return pre_article


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


def calculate_weight(words):
    word_weight_dict = {}
    for i, word_list in enumerate(words):
        if i == 0:
            for word in word_list:
                if word not in stopwords:
                    if word not in word_weight_dict.keys():
                        word_weight_dict[word] = 1
                    else:
                        word_weight_dict[word] += 1
        else:
            for word in word_list:
                if word not in stopwords:
                    if word not in word_weight_dict.keys():
                        word_weight_dict[word] = round((5 - i) * 0.1, 3)
                    else:
                        word_weight_dict[word] += round((5 - i) * 0.1, 3)
    return word_weight_dict


def count_entity(words):
    word_count_dict = {}
    for word_list in words:
        for word in word_list:
            if word not in stopwords and word: # per replace之后可能为空
                if word not in word_count_dict.keys():
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] += 1
    return word_count_dict


def phrase_deduplication(phrases):
    # 除了对短语进行去重外，应该把一些没有用的核心词（动词）去掉
    # 返回的短语要处理一下，其中的宾语可以是phase对象，从而进行去重处理, 也不用，只需对短语分割就好
    deduplicated_phrases = []
    for i, p1 in enumerate(phrases):
        # print(len(phrases))
        # print("Number: %d, %s" % (i, p1.get()))
        choice = 0
        # for j, p2 in enumerate(phrases[i+1:]):
        # for j, p2 in enumerate(phrases):
        a = [p2 for p2 in phrases if p2 != p1]
        # print(len(a))
        for p2 in a:
            # if len(p1.get_verbs()) == 1:
            #     if p1.get_verbs()[0] in p2.get_verbs() and len
            # print(p1.predicate)
            # print(p1.object)
            # print(p1.get_verbs())
            # print(set(p1.get_verbs()))
            # print(set(p2.get_verbs()))

            if len(list(set(p1.get_verbs()).intersection(p2.get_verbs()))) > 0: # 有交集
                if len(p1.get_verbs()) > len(p2.get_verbs()):
                    choice = 1
                if len(p1.get_verbs()) == len(p2.get_verbs()):
                    if len(p1.subject)+len(p1.object) >= len(p2.subject)+len(p2.object): # 这有问题
                        choice = 1
                    else:
                        choice = 0
                if len(p1.get_verbs()) < len(p2.get_verbs()):
                    choice = 0
            else:
                choice = 1
            # print("Choice: %d" % choice)

            if len(p1.subject)+len(p1.object) < 2:
                choice = 0
            if choice == 0: # 一旦找到更好的就break
                break
        # 还想到一办法就是，维持一个[1,1,1,1,1...]的列表，可以替代的置0，当第一个for遍历的时候，如果是0，则跳过。

        if choice == 1:
            deduplicated_phrases.append(p1)
    return deduplicated_phrases


