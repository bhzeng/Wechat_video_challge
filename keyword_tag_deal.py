import pandas as pd
from collections import defaultdict
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

if __name__ == "__main__":
    feed = pd.read_csv('/home/zengbuhui//data/wechat_algo_data1/feed_info.csv')

    # keywords and tags do TF-IDF action with sklearn
    # 8: manual_keyword_list, 9: machine_keyword_list, 10: manual_tag_list, 11: machine_keyword_list
    feed_keyword = feed['manual_keyword_list'] + ';' + feed['machine_keyword_list']
    feed_tag = feed['manual_tag_list'] + ';' + feed['machine_tag_list']
    # deal with NaN, some sample without keyword and tag
    feed_keyword = list(feed_keyword.fillna("0", ))
    feed_tag = list(feed_tag.fillna("0", ))
    sents = [feed_keyword[0], feed_keyword[1], feed_keyword[2]]
    print(sents)
    vectorizer = CountVectorizer()
    corpus_vector = vectorizer.fit_transform(sents)
    words = vectorizer.get_feature_names()
    print(words)
    # print(corpus_vector)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(corpus_vector)
    # tf_idf 表示第n个词在第m篇文档的tf_idf值。提取前k个关键词只需要将矩阵按行的值从大到小排序取前几个即可。
    weight = tf_idf.toarray()
    print(weight)
    for i in range(len(sents)):
        data = {'word': words, "tf_idf": weight[i]}
        df = pd.DataFrame(data)
        print(df)
        df.sort_values(by="tf_idf", ascending=False)
        print(df)
        w = df['word'].tolist()
        print(w)