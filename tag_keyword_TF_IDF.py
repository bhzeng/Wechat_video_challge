import pandas as pd
from collections import defaultdict
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_K_words_from_TFIDF(sents, k):
    vectorizer = CountVectorizer(max_features=k)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(sents))
    weight = tf_idf.toarray()
    # 词汇表中前 K 个单词在语句中的tf_idf值
    return weight
    # # 注意这个由于脱敏处理的原因，数字代表了词语，数字间使用 ;分割
    # # 所以先把单词分割成列表
    # for i in range(len(sents)):
    #     sents[i] = list(sents[i].split(";"))
    #
    # # 1: 统计单词的词频 TF
    # doc_frequency = defaultdict(int)
    # for word_list in sents:
    #     for iword in word_list:
    #         doc_frequency[iword] += 1
    # word_tf = {}
    # for iword in doc_frequency:
    #     word_tf[iword] = doc_frequency[iword] / sum(doc_frequency.values())
    #
    # # 2: 统计每个单词的 IDF 值
    # doc_nums = len(sents)
    # word_idf = {}
    # word_doc = defaultdict(int)  # 存储包含该单词的文档数
    # for iword in doc_frequency:
    #     for word_list in sents:
    #         if iword in word_list:
    #             word_doc[iword] += 1
    # for iword in doc_frequency:
    #     word_idf[iword] = math.log(doc_nums / (word_doc[iword]+1) )
    #
    # # 3: 计算每个单词的 TF*IDF 值
    # word_tf_idf = {}
    # for iword in doc_frequency:
    #     word_tf_idf[iword] = word_tf[iword] * word_idf[iword]
    #
    # # 4: 根据TF_IDF值筛选每文档中最大 k 个不重复单词作为代表
    # ans = []
    # for word_list in sents:
    #     TFIDF_list = []
    #     # 取负数时排序中越大越前
    #     for iword in word_list:
    #         TFIDF_list.append(-1*word_tf_idf[iword])
    #     # 根据TF_IDF的排序索引来给原序列排序，选择最大的 k 个
    #     res = [i for _,i in sorted(zip(TFIDF_list, word_list))]
    #     n = len(res)
    #     while n < k:
    #         res.append('0')
    #         n += 1
    #     ans.append(res[:k])
    # return ans

if __name__ == "__main__":
    feed = pd.read_csv('/home/zengbuhui//data/wechat_algo_data1/feed_info.csv')

    # keywords and tags do TF-IDF action with sklearn
    # 8: manual_keyword_list, 9: machine_keyword_list, 10: manual_tag_list, 11: machine_keyword_list
    feed_keyword = feed['manual_keyword_list'] + ';' + feed['machine_keyword_list']
    feed_tag = feed['manual_tag_list'] + ';' + feed['machine_tag_list']
    # deal with NaN, some sample without keyword and tag
    feed_keyword = list(feed_keyword.fillna("0", ))
    feed_tag = list(feed_tag.fillna("0", ))
    ans_keyword = get_K_words_from_TFIDF(feed_keyword, 10)
    ans_tagword = get_K_words_from_TFIDF(feed_tag, 10)
    print(ans_tagword[:10])
