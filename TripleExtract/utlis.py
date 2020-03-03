

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