from pyhanlp import HanLP
from HotSpotServer.component.event_analyze.doc import Phrase


def extract_subject(word, words):
    """
    根据确定的主谓关系，从句法分析树中抽取主语实体
    :param word: 与核心词为主语关系的词
    :param words: 句法分析树
    :return: 主语(可能附带主语定语)的字符串
    """
    sub = word.LEMMA
    # 从当前词到句子首部，检索可能存在的主语的定语
    for i in range(word.ID-2, -1, -1):
        # ??? 在找定中关系时，要不要考虑距离，因为可能一个名词没有定中关系的词，这样仍需要从该词位置反遍历到句子开头
        if words[i].HEAD.ID == word.ID and words[i].DEPREL == '定中关系':
            sub = words[i].LEMMA + sub
            break
    return sub


def extract_object(word, words):
    """
    根据确定的谓宾关系，从句法分析树中抽取宾语实体，有可能宾语是动词，要递归调用
    :param word: 与核心词为宾语关系的词
    :param words: 句法分析树
    :return: 宾语(可能附带主语定语)的字符串
    """
    # 动宾关系的词有可能是个动词，假如是动词还要另作处理
    obj = word.LEMMA
    # 从当前词到句子首部，检索可能存在的宾语的定语
    for i in range(word.ID-2, -1, -1):
        if words[i].HEAD.ID == word.ID and words[i].DEPREL == '定中关系':
            obj = words[i].LEMMA + obj
            break
    # 如果宾语是动词，在递归查找宾语的宾语，直到宾语不是动词
    if word.CPOSTAG == 'v':
        for j in range(word.ID, len(words)):
            if words[j].HEAD.ID == word.ID and words[j].DEPREL == '动宾关系':
                # 递归寻找宾语
                obj = obj + ' ' + extract_object(words[j], words)
    return obj


def extract_event_detail(word, words):
    """
    根据句子句法分析得到的"核心词"，提取"核心词"相关的主语和宾语（也就是event），一般"核心词"为动词，
    但是可能存在与"核心词"并列的词，因此可能要递归调用，一个Phrase实例代表一个event。
    :param word: 核心词/核心词并列词
    :param words: 句法分析树
    :return: Phrase实例列表
    """
    subs = []
    objs = []
    phrase = Phrase()
    phrases = []
    for item in words:
        if item.HEAD.ID == word.ID:
            # 提取主语
            if item.DEPREL == '主谓关系':
                # print("sub-pre---"+item.LEMMA)
                subs.append(extract_subject(item, words))
                # print("subs: "+str(subs))
            # 提取宾语
            if item.DEPREL == '动宾关系':
                # print("pre-obj---" + item.LEMMA)
                objs.append(extract_object(item, words))
                # print("objs: "+str(objs))
            # 并列动词，递归调用
            if item.DEPREL == '并列关系':
                # print("pre-并列---" + item.LEMMA)
                phrases.extend(extract_event_detail(item, words))

    # 主语/宾语必须包含一个
    if len(subs) != 0 or len(objs) != 0:
        phrase.subject = subs
        phrase.object = objs
        phrase.predicate = word.LEMMA
        phrases.append(phrase)

    # if len(subs) != 0 and len(objs) != 0:
    #     phrase.subject = subs
    #     phrase.object = objs
    #     phrase.predicate = word.LEMMA
    #     # phrase.show()
    #     phrases.append(phrase)

    return phrases


def get_sentence_event_detail(sentence, n=-1):
    """
    对句子通过"依存句法分析"提取"主谓宾"短语
    :param sentence: 输入的句子
    :param n: 指定返回句子Phrase的个数
    :return: Phrase实例列表
    """
    phrases = []
    words = HanLP.parseDependency(sentence).word
    for word in words:
        if word.DEPREL == '核心关系' and word.CPOSTAG == 'v':  # 核心关系词需要是动词，要不然没有提取的必要
            if n == -1:
                phrases.extend(extract_event_detail(word, words))
            else:
                # phrases.extend(extract_event_detail(word, words)[0])
                phrases.append(extract_event_detail(word, words)[0])
    return phrases


def get_article_event_detail(prep_article, sentence_type='original', sentence_count=4):
    """
    对每篇文章提取"title"和"评分得到的中心句子通过"依存句法分析"提取"主谓宾"短语
    :param prep_article: PreprocessArticle类的实例
    :param sentence_type: 待提取的句子的排序方法，取值为 original/score, 分别代表 文章原始句子顺序，文章句子评分排序
    :param sentence_count: 中心句的个数，默认为4，如果为0，这只对title进行提取
    :return: Phrase实例列表
    """
    phrases = []
    # 文章title
    words = HanLP.parseDependency(prep_article.title).word
    # print(prep_article.title)
    # for word in words:
    #     print(word.ID, word.DEPREL, word.LEMMA, word.CPOSTAG)
    for word in words:
        if word.DEPREL == '核心关系' and word.CPOSTAG == 'v':  # 核心关系词需要是动词，要不然没有提取的必要
            # print("-----"+word.LEMMA)
            phrases.extend(extract_event_detail(word, words))
    # 文章句子
    if sentence_count > 0:
        # 文章前n个句子
        if sentence_type == 'original':
            for i, sentence in enumerate(prep_article.sentences):
                if i < sentence_count:
                    words = HanLP.parseDependency(sentence.text).word
                    for word in words:
                        if word.DEPREL == '核心关系' and word.CPOSTAG == 'v':
                            phrases.extend(extract_event_detail(word, words))
        # 文章得分降序前n个句子
        if sentence_type == 'score':
            for i, idx in enumerate(prep_article.descend_sentence_index):
                if i < sentence_count:
                    words = HanLP.parseDependency(prep_article.sentences[idx].text).word
                    for word in words:
                        if word.DEPREL == '核心关系' and word.CPOSTAG == 'v':
                            phrases.extend(extract_event_detail(word, words))

    phrases.reverse()   # 因为extract_event_detail()是递归调用
    # print(len(phrases))
    return phrases
