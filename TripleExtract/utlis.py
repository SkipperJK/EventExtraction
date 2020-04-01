import re

def get_WordsDictOfRole(words, postags, word_start_index, word_end_index, ):
    """
    解析语义角色中包含的词和词的词性
    :param words:
    :param postags:
    :param word_start_index:
    :param word_end_index:
    :return:
    """
    role_words_dict = {}
    for word_index in range(word_start_index, word_end_index+1):
        # 过滤掉词性为 u, wp（标点符号）, ws（外语词）, x
        if postags[word_index][0] not in ['w', 'u', 'x']:
            role_words_dict[words[word_index]] = postags[word_index]

    return role_words_dict


def split_sentence(content):
    content = re.sub('([。！？\?])([^”’])', r"\1\n\2", content)  # 单字符断句符
    content = re.sub('(\.{6})([^”’])', r"\1\n\2", content)  # 英文省略号
    content = re.sub('(\…{2})([^”’])', r"\1\n\2", content)  # 中文省略号
    content = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', content)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    content = content.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return content.split("\n")