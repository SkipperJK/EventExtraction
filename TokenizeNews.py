import pkuseg
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
    title = ''
    seg_pos_title = []
    sentences = []
    topic_words = []
    descend_sentence_index = []
    max_sentence_original_socre = 0
    token_weight_dic = {}
    #     seg_pos_content = []
    #     seg_pos_all_text = []
    event = None

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


def preprocessing(news_list):
    """
    对News进行预处理，对文本进行分词和词性标注
    这里的分词,没有去掉stopwords,也没有去掉标点符号

    Args:
        news_list: 原始的新闻list

    Return：
        预处理之后的新闻list
    """
    tokenizeNews_list = []
    for news in news_list:
        tokenizeNews = TokenizeNews()
        tokenizeNews.title = news.title
        tokenizeNews.seg_pos_title = seg_pos.cut(news.title)
        sentences = SentenceSplitter.split(news.content)  # Using ltp split sentence
        sentences = [x for x in sentences if x != '']
        tokenizeNews.sentences = []  # 这里如果不这样,seg_pos_sentences的个数会一直增加， #感觉是TokenizeNews() 得到的变量tokenizeNews成员变量属性值没有重置
        for idx, x in enumerate(sentences):
            sentence = Sentence()
            sentence.text = x
            sentence.location = idx
            sentence.seg_pos = seg_pos.cut(x)
            tokenizeNews.sentences.append(sentence)
        tokenizeNews_list.append(tokenizeNews)

    print(len(tokenizeNews_list))
    return tokenizeNews_list


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
