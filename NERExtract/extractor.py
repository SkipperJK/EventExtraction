import logging
from NERExtract.ner_extract import get_ner_weight_dict, get_dnn_entity_count_dict
from NERExtract.spo_extract import get_hanlp_spo_weight_dict, get_ltp_spo_weight_dict
from NERExtract.phrase_extract import get_article_event_detail
from NERExtract.utils import article_preprocessing, tfidf, phrase_deduplication
from NERExtract.doc import PreprocessArticle, Sentence, Event
from TripleExtract.triple_extract_bak import TripleExtractor
from pyltp import SentenceSplitter

logger = logging.getLogger(__name__)
extractor = TripleExtractor()


def extract_event(articles, sentence_count=4, method='original', dp_method='hanlp'):
    """
    根据相应的事件模型，对文章进行事件提取
    响应时间：500ms内
    :param article: 待提取的文章
    :param n: 除了文章title之外，还要进行分析的文章中句子的个数
    :param method:
    :return: 提取的事件
    """
    # TODO 汲少康

    events = []

    if method not in ['score', 'original']:
        raise Exception("The parameter \"method\" ERROR!")
    if dp_method not in ['hanlp', 'ltp']:
        raise Exception("The parameter \"dp_method\" ERROR!")

    if method == 'score':
        events = extract_event_by_score(articles, sentence_count, dp_method)
    elif method == 'original':
        events = extract_event_by_original(articles, sentence_count, dp_method)

    return events


def extract_event_by_score(articles, sentence_count=4, dp_method='hanlp'):

    prep_articles = []
    for article in articles:
        prep_articles.append(article_preprocessing(article))

    print("calc tfidf")
    term_document_matrix = [pre_article.all_text() for pre_article in prep_articles]
    tfidf_matrix, vocabulary = tfidf(term_document_matrix)
    tfidf_array = tfidf_matrix.toarray()

    print("record tfidf")
    print("extract topic words")
    for i, prep_article in enumerate(prep_articles):
        prep_article.record_token_weight(i, tfidf_array, vocabulary)
        prep_article.extract_topicwords(i, tfidf_array, vocabulary)

    print("calc sentence score")
    for prep_article in prep_articles:
        prep_article.calculate_sentence_original_score()
    for prep_article in prep_articles:
        prep_article.calculate_sentence_score()
    for prep_article in prep_articles:
        prep_article.sentence_sort()

    print("event extraction")
    for i, prep_article in enumerate(prep_articles):
        print(i, end='\t')
        if i % 30 == 0:
            print()

        event = Event()
        # 使用神经网络NER模型提取命名实体，并计算每个实体的权重
        [event.date_count_dict, event.per_count_dict, event.loc_count_dict, event.org_count_dict] = get_ner_weight_dict(
            prep_article, sentence_type='score', sentence_count=sentence_count)
        # 使用依存句法分析，提取Sub,Predicate,Obj，并计算每个的权重
        if dp_method == 'hanlp':    # 使用pyhanlp进行句法分析
            [event.who_count_dict, event.whom_count_dict, event.predicate_count_dict] = get_hanlp_spo_weight_dict(
                prep_article, sentence_type='score', sentence_count=sentence_count)
        elif dp_method == 'ltp':    # 使用pyltp进行句法分析
            [event.who_count_dict, event.whom_count_dict, event.predicate_count_dict] = get_ltp_spo_weight_dict(
                prep_article, sentence_type='score', sentence_count=sentence_count)
        # 使用依存句法分析，提取SPO短语
        phrases = get_article_event_detail(prep_article, sentence_type='score', sentence_count=sentence_count)
        event.phrases = phrase_deduplication(phrases)

        prep_article.event = event

    event_list = []
    for prep_article in prep_articles:
        event_list.append(prep_article.event)
    return event_list


def extract_event_by_original(articles, sentence_count=4, dp_method='hanlp'):

    prep_article_list = []
    for article in articles:
        prep_article_list.append(article_preprocessing(article))

    print("event extraction")
    triples = []
    for i, prep_article in enumerate(prep_article_list):
        print(i, end='\t')
        if i % 30 == 0 and i != 0:
            print()

        svos = extractor.triples_main(prep_article.title)
        # print('svos', svos)
        if not svos:
            phrases = get_article_event_detail(prep_article, sentence_type='score', sentence_count=sentence_count)
            for phrase in phrases:
                svos.append(phrase.get())
        triples.append(svos)
    print()
    return triples
'''
        event = Event()
        # # 使用神经网络NER模型提取命名实体，并计算每个实体的权重
        # [event.date_count_dict, event.per_count_dict, event.loc_count_dict, event.org_count_dict] = get_ner_weight_dict(
        #     prep_article, sentence_type='original', sentence_count=sentence_count)
        # # 使用依存句法分析，提取Sub,Predicate,Obj，并计算每个的权重
        # if dp_method == 'hanlp':  # 使用pyhanlp进行句法分析
        #     [event.who_count_dict, event.whom_count_dict, event.predicate_count_dict] = get_hanlp_spo_weight_dict(
        #         prep_article, sentence_type='original', sentence_count=sentence_count)
        # elif dp_method == 'ltp':  # 使用pyltp进行句法分析
        #     [event.who_count_dict, event.whom_count_dict, event.predicate_count_dict] = get_ltp_spo_weight_dict(
        #         prep_article, sentence_type='original', sentence_count=sentence_count)
        # 使用依存句法分析，提取SPO短语
        phrases = get_article_event_detail(prep_article, sentence_type='score', sentence_count=sentence_count)
        # print("Receive length" + str(len(phrases)))
        # event.phrases = phrase_deduplication(phrases)     # 去重有问题
        # print("Deduplicate length" + str(len(event.phrases)))
        event.phrases = phrases

        prep_article.event = event

    event_list = []
    for prep_article in prep_article_list:
        event_list.append(prep_article.event)
    return event_list
    '''


class TestExtractor(TestCase):
    def test_log(self):
        logger.info("Testing logger...")

    def test_ee(self):
        class TestArticle:
            def __init__(self, title, content, topic, time):
                self.title = title
                self.content = content
                self.topic = topic
                self.time = time

        logger.info("Testing  Event Extraction...")
        import pandas as pd
        df = pd.read_csv("data/Sina.737news.csv")
        df = df[df.topic != "其他"].reset_index()
        articles = [TestArticle(df.title[i], df.text[i], df.topic[i], df.time[i]) for i in range(df.shape[0])]
        # print(len(articles))
        event_list = extract_event(articles, sentence_count=0, method='original', dp_method='hanlp')

        for i, event in enumerate(event_list):
            print("第 %d 个事件：" % i)
            print("Event detail: ")
            # print("\tWhen: %s" % (event_list[i].date))
            # print("\tWhere: %s" % (event_list[i].location))
            print("\tdate_count_dict: {}".format(event_list[i].date_count_dict))
            print("\tper_count_dict: {}".format(event_list[i].per_count_dict))
            print("\tloc_count_dict: {}".format(event_list[i].loc_count_dict))
            print("\torg_count_dict: {}".format(event_list[i].org_count_dict))
            print("\twho_count_dict: {}".format(event_list[i].who_count_dict))
            print("\twhom_count_dict: {}".format(event_list[i].whom_count_dict))
            print("\tpredicate_count_dict: {}".format(event_list[i].predicate_count_dict))
            for phrase in event.phrases:
                phrase.show()
                # print(phrase.show())
            # print("\tloc_count_dict_dnn: {}".format(event_list[i].loc_count_dict_dnn))

        # fp = open("HotSpotServer/component/event_analyze/event_log.txt", "w")
        # for i, event in enumerate(event_list):
        #     fp.write("第 %d 个事件：\n" % i)
        #     fp.write("Event detail: \n")
        #     fp.write("\tWhen: %s\n" % (event_list[i].date))
        #     fp.write("\tWhere: %s\n" % (event_list[i].location))
        #     fp.write("\twho_count_dict: {}\n".format(event_list[i].who_count_dict))
        #     fp.write("\twhom_count_dict: {}\n".format(event_list[i].whom_count_dict))
        #     fp.write("\tpredicate_count_dict: {}\n".format(event_list[i].predicate_count_dict))
        #     fp.write("\tloc_count_dict: {}\n".format(event_list[i].loc_count_dict))
        #     fp.write("\tdate_count_dict: {}\n".format(event_list[i].date_count_dict))
        #     fp.write("\tper_count_dict: {}\n".format(event_list[i].per_count_dict))
        #     fp.write("\tloc_count_dict_dnn: {}\n".format(event_list[i].loc_count_dict_dnn))
        #     fp.write("\torg_count_dict: {}\n".format(event_list[i].org_count_dict))
        # fp.close()


    def test_ner(self):
        # serializer = ArticleSerializer(Article.objects.all(), many=True)    # 太慢了，似乎是一次全部读取到内存中
        # articles = serializer.data
        # for article in articles:
        #     print(article.title)
        # print(len(article))
        from pymongo import MongoClient
        import json
        MONGODB_DATABASE_NAME = 'Sina'
        MONGODB_HOST = '10.141.212.160'
        MONGODB_PORT = 27017
        MONGODB_ARTICLE_COLLECTION = 'article20190413'  # articleTest
        conn = MongoClient(MONGODB_HOST, MONGODB_PORT)
        db = conn[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_ARTICLE_COLLECTION]
        all_articles = collection.find({})
        print(all_articles.count())
        i = 0

        per_dict = {}
        for article in all_articles:
            i = i + 1
            print(i, end=',')

            prep_article = PreprocessArticle()
            prep_article.title = article['title']
            sentences = SentenceSplitter.split(article['content'])  # Using ltp split sentence
            sentences = [x for x in sentences if x != '']
            for idx, x in enumerate(sentences):
                sentence = Sentence()
                sentence.text = x
                prep_article.sentences.append(sentence)

            tmp = get_dnn_entity_count_dict(prep_article, 'per', sentence_type='original', sentence_count=1000)
            for key in tmp.keys():
                if key in per_dict:
                    per_dict[key] += tmp[key]
                else:
                    per_dict[key] = tmp[key]

            # if i == 1:
            #     break
        data = json.dumps(per_dict, ensure_ascii=False)
        # print(json.dumps(per_dict, ensure_ascii=False))
        with open('./celebrity_name_list.txt', 'w') as fp:
            fp.write(data)
