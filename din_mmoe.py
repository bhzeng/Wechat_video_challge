import tensorflow as tf
import deepctr
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from deepctr.inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer
from deepctr.layers.utils import concat_func, NoMask, combined_dnn_input, reduce_sum

def DIN_MMoE(dnn_feature_columns, history_feature_list, num_experts=3, dnn_use_bn=False,
             dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
             expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64, ), gate_dnn_hidden_units=(),
             att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
             task_types=('binary', "binary"), task_names=('ctr', 'ctcvr')):

    # # 0 来自 mmoe 的判断
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if num_experts <= 1:
        raise ValueError("num_experts must be greater than 1")

    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    # # 1：这部分来自于DIN，去掉最后的利用DNN的output部分.
    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    dnn_input_emb_list += sequence_embed_list

    keys_emb = concat_func(keys_emb_list, mask=True)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list, mask=True)
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([query_emb, keys_emb])

    deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist])
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    mmoe_input = combined_dnn_input([deep_input_emb], dense_value_list)

    # # 2 来自MMoE的主要部分，去掉输入层
    # build expert layer
    expert_outs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                             name='expert_' + str(i))(mmoe_input)
        expert_outs.append(expert_network)

    expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim

    mmoe_outs = []
    for i in range(num_tasks):  # one mmoe layer: nums_tasks = num_gates
        # build gate layers
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                         name='gate_' + task_names[i])(mmoe_input)
        gate_out = tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax',
                                         name='gate_softmax_' + task_names[i])(gate_input)
        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate multiply the expert
        gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                 name='gate_mul_expert_' + task_names[i])([expert_concat, gate_out])
        mmoe_outs.append(gate_mul_expert)

    task_outs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(mmoe_out)

        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
