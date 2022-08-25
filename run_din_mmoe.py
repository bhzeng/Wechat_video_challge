import numpy as np
from din_mmoe import DIN_MMoE
import os
import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.decomposition import PCA
from evaluation import evaluate_deepctr
from time import time

if __name__ == "__main__":
    # GPU相关设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    epochs = 2
    batch_size = 4
    embedding_dim = 8

    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    dense_features = ['videoplayseconds', ]
    behavior_feature_list = ["feedid", ]

    # read data
    print("start read data: ")
    data = pd.read_csv('/home/zengbuhui/Project/Wechat_video/wechat_algo_data1/user_action.csv')
    feed = pd.read_csv('/home/zengbuhui/Project/Wechat_video/wechat_algo_data1/feed_info.csv')
    feed_embedding = pd.read_csv('/home/zengbuhui/Project/Wechat_video/wechat_algo_data1/feed_embeddings.csv')
    test = pd.read_csv('/home/zengbuhui/Project/Wechat_video/wechat_algo_data1/test_a.csv')
    print("data finish! ")

    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')

    # # PCA for feed_embeddings, from 512 to 64
    # print("PCA for feed_embedding: ")
    # embedding_data = []
    # for i in range(len(feed_embedding)):
    #     dl = list(map(float, feed_embedding.iloc[i, 1].split()))
    #     embedding_data.append(dl)
    # embedding_data = np.array(embedding_data)
    # print(embedding_data.shape)
    # pca = PCA(n_components=64)
    # low_dim_data = pca.fit_transform(embedding_data)
    # print(low_dim_data.shape)
    # print("PCA finish! ")

    # merge feed to user behavior data
    data = data.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')
    test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')

    # 1.fill nan dense_feature and do simple Transformation for dense features
    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )

    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)

    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    # 对数据集根据用户id进行分组，根据日期进行排序。
    data_g = data.groupby("userid", sort=False).apply(lambda x: x.sort_values("date_", ascending=True)).reset_index(drop=True)

    # 把数据集划分为测试集和验证集
    train = data_g[data_g['date_'] == 13]
    print(len(train))
    val = data_g[data_g['date_'] == 14]  # 第14天样本作为验证集

    # 构建 train 中 feeid 历史行为队列
    train_h = data_g[["userid", "feedid", "date_"]].merge(train["userid"], how='right', on='userid')
    train_h = train_h[train_h['date_'] < 13]
    userid_l = train_h["userid"].drop_duplicates(keep='first').tolist()
    print(len(userid_l))
    # groupby之后的数据结构是由元祖构成的（分组名，数据块），数据块也是dataframe结构。
    userid_th = train_h.groupby(by="userid").apply(lambda x: x['feedid'].tolist())
    userid_t = userid_th.tolist()
    userid_D = pd.DataFrame()
    userid_D['userid'] = userid_l
    userid_D['feedid_his'] = userid_t
    train = train.merge(userid_D, how='left', on='userid')
    print('train.columns', train.columns.tolist())

    # 构建 val 中 feeid 历史行为队列
    val_h = data_g[["userid", "feedid", "date_"]].merge(val["userid"], how='right', on='userid')
    val_h = val_h[val_h['date_'] < 13]
    userid_l = val_h["userid"].drop_duplicates(keep='first').tolist()
    print(len(userid_l))
    # groupby之后的数据结构是由元祖构成的（分组名，数据块），数据块也是dataframe结构。
    userid_th = val_h.groupby(by="userid").apply(lambda x: x['feedid'].tolist())
    userid_t = userid_th.tolist()
    userid_D = pd.DataFrame()
    userid_D['userid'] = userid_l
    userid_D['feedid_his'] = userid_t
    val = val.merge(userid_D, how='left', on='userid')
    print('test.columns', val.columns.tolist())
    #
    # # 构建 train 中 authorid 历史行为队列
    # train_h = data_g[["userid", "authorid", "date_"]].merge(train["userid"], how='right', on='userid')
    # train_h = train_h[train_h['date_'] < 13]
    # userid_l = train_h["userid"].drop_duplicates(keep='first').tolist()
    # print(len(userid_l))
    # # groupby之后的数据结构是由元祖构成的（分组名，数据块），数据块也是dataframe结构。
    # userid_th = train_h.groupby(by="userid").apply(lambda x: x['authorid'].tolist())
    # userid_t = userid_th.tolist()
    # userid_D = pd.DataFrame()
    # userid_D['userid'] = userid_l
    # userid_D['authorid_his'] = userid_t
    # train = train.merge(userid_D, how='left', on='userid')
    # print('train.columns', train.columns.tolist())
    #
    # # 构建 val 中 feeid 历史行为队列
    # val_h = data_g[["userid", "authorid", "date_"]].merge(val["userid"], how='right', on='userid')
    # val_h = val_h[val_h['date_'] < 13]
    # userid_l = val_h["userid"].drop_duplicates(keep='first').tolist()
    # print(len(userid_l))
    # # groupby之后的数据结构是由元祖构成的（分组名，数据块），数据块也是dataframe结构。
    # userid_th = val_h.groupby(by="userid").apply(lambda x: x['authorid'].tolist())
    # userid_t = userid_th.tolist()
    # userid_D = pd.DataFrame()
    # userid_D['userid'] = userid_l
    # userid_D['authorid_his'] = userid_t
    # val = val.merge(userid_D, how='left', on='userid')
    # print('test.columns', val.columns.tolist())

    train['feedid_his'] = train['feedid_his'].fillna(0, )
    train_his_feedid = train['feedid_his'].tolist()
    for i in range(len(train_his_feedid)):
        if train_his_feedid[i] == 0:
            train_his_feedid[i] = [0]*40
        if len(train_his_feedid[i]) > 40:
            train_his_feedid[i] = train_his_feedid[i][-40:]
        elif len(train_his_feedid[i]) < 40:
            train_his_feedid[i].extend([0] * (40 - len(train_his_feedid[i])))
    train_his_feedid = np.array(train_his_feedid)
    train['feedid_his'] = train_his_feedid

    val['feedid_his'] = val['feedid_his'].fillna(0, )
    his_feedid = val['feedid_his'].tolist()
    for i in range(len(his_feedid)):
        if his_feedid[i] == 0:
            his_feedid[i] = [0, 0]
        if len(his_feedid[i]) > 40:
            his_feedid[i] = his_feedid[i][-40:]
        elif len(his_feedid[i]) < 40:
            his_feedid[i].extend([0] * (40 - len(his_feedid[i])))
    his_feedid = np.array(his_feedid)
    val['feedid_his'] = his_feedid

    # 输入特征
    feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    feature_columns += [VarLenSparseFeat(SparseFeat('feedid_his', vocabulary_size=8, embedding_dim=embedding_dim,
                                                    embedding_name='feedid'), maxlen=40),]
    feature_names = get_feature_names(feature_columns)
    print(feature_names)

    # 数据集的构建
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    train_model = DIN_MMoE(feature_columns, behavior_feature_list, task_types=['binary', 'binary', 'binary', 'binary'],
                           task_names=target)
    train_model.compile("adagrad", loss='binary_crossentropy')

    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)