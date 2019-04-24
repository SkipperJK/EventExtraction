#coding=utf-8
from News import *
from preprocessing import *
from TokenizeNews import *
from Sentence import *
from Event import *





news_list = read_mongo("Sina", "737news")
tokenizeNews_list = preprocessing(news_list)

all_texts = [tokenizeNews.all_text() for tokenizeNews in tokenizeNews_list]
all_tfidf_matrix, all_vocabulary = tfidf(all_texts)
all_tfidf_array = all_tfidf_matrix.toarray() # transfer scipy.crs.crs_matirx to numpy narray


record_token_weight(tokenizeNews_list, all_tfidf_array, all_vocabulary)

extract_news_topicwords(tokenizeNews_list, all_tfidf_array, all_vocabulary, N = 8)


calculate_sentence_original_score(tokenizeNews_list)

calculate_sentence_score(tokenizeNews_list)

sentence_sort(tokenizeNews_list)

find_location(tokenizeNews_list, N = 5)

find_date(tokenizeNews_list, N = 5)

# extract_arg(tokenizeNews_list)
