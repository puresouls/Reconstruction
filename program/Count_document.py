# coding:utf-8
__author__ = 'cheng'

# 根据特征统计文档个数,并且过滤一个查询下文档在其他查询下出现过20次以上的数据


import pandas as pd


def document_number(train_file, validation_file, test_file):
    feature_list = [12, 13, 14, 15, 16, 127, 128, 129, 130, 131, 132, 133, 134]
    training_data = pd.read_table(train_file, sep=' ', header=None)
    validation_data = pd.read_table(validation_file, sep=' ', header=None)
    test_data = pd.read_table(test_file, sep=' ', header=None)
    training_data = training_data.append(validation_data.iloc[:], ignore_index=True)
    training_data = training_data.append(test_data.iloc[:], ignore_index=True)
    data = training_data.groupby(feature_list).count().reset_index()
    data[10000] = range(1, len(data)+1)
    feature_list.append(10000)
    data = data[feature_list]
    data_set = pd.merge(training_data, data, how='left', on=feature_list[0:13])
    merge_list = [0, 1, 12, 13, 14, 15, 16, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 10000]
    data_set = data_set[merge_list]
    return data_set



train_file = "../MSLR-WEB10K/Fold1/train.txt"
validation_file = "../MSLR-WEB10K/Fold1/vali.txt"
test_file = "../MSLR-WEB10K/Fold1/test.txt"
data_set = document_number(train_file, validation_file, test_file)

temp_doc = data_set.groupby(['10000']).count()
temp_doc = temp_doc[temp_doc['1'] > 20]
qid_doc = data_set[['1', '10000']]
temp_data = pd.merge(temp_doc, qid_doc,  left_index=True, right_on='10000', how='left')
temp_data = temp_data[['1_y', '10000']]
temp_qid = temp_data.groupby(['1_y']).count().reset_index()[['1_y']]
data_part = pd.merge(data_set, temp_qid,  left_on='1', right_on='1_y')

data_part.to_csv("../data/data_part_beyond_20.csv")