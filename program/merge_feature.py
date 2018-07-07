# coding:utf-8
__author__ = 'cheng'

# 这个脚本是把重构完成的了点击特征与原数据集进行对接

import numpy as np
import pandas as pd
import os


def process(column):
    return map(lambda x: x.split(':')[-1], column)


def merge_data(query_number, group_number):

    data_part = pd.read_csv("../data/data_part_beyond_20.csv")
    merge_feats = ['0', '1', '12', '13', '14', '15', '16', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '10000']
    data_part = data_part[merge_feats]

    df = pd.DataFrame(columns=[])
    for i in range(1, group_number+1):
        # data_file = ("../click_mf/%d.txt" % i)
        data_file = ("../b-nmf/group_%d/result/%d.txt" % (query_number, i))
        data_mf = pd.read_csv(data_file, sep=' ', header=None)
        group = pd.read_csv("../data/group_%d/group_%d.csv" % (query_number, i))
        group.set_index(['1'], inplace=True)
        data_mf.set_index(group.index, inplace=True)
        data_mf.columns = group.columns
        data_mf = data_mf.reset_index()
        data = pd.melt(data_mf, id_vars=['1'], var_name='10000', value_name='click_mf')
        df = df.append(data.iloc[:], ignore_index=True)

    df['10000'] = df['10000'].astype(int)
    data_set = pd.merge(data_part, df, on=['1', '10000'], how='inner')
    data_set['click_mf'] = data_set['click_mf'].fillna(0)

    mer_feats = ['0', '1', '12', '13', '14', '15', '16', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', 'click_mf']
    data_set = data_set[mer_feats]
    feats = [0, 1, 12, 13, 14, 15, 16, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 'click_mf']
    data_set.columns = [0, 1, 12, 13, 14, 15, 16, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 'click_mf']
    click_fea = list(data_set['click_mf'])
    click_fea = map(lambda x: "137:"+str(x), click_fea)
    data_set['click_fea'] = click_fea
    data_set = data_set.drop('click_mf', 1)
    # 以上部分对超过20篇重复的原始数据集加入了重构的点击特征

    for i in range(1, 6):
        training_set = pd.read_csv("../MSLR-WEB10K/Fold%d/train.txt" % i, sep=' ', header=None)
        validation_set = pd.read_csv("../MSLR-WEB10K/Fold%d/vali.txt" % i, sep=' ', header=None)
        test_set = pd.read_csv("../MSLR-WEB10K/Fold%d/test.txt" % i, sep=' ', header=None)
        training_set = training_set.drop(138, 1)
        validation_set = validation_set.drop(138, 1)
        test_set = test_set.drop(138, 1)
        training_set = pd.merge(training_set, data_set, how='inner', on=feats[0:-1])
        validation_set = pd.merge(validation_set, data_set, how='inner', on=feats[0:-1])
        test_set = pd.merge(test_set, data_set, how='inner', on=feats[0:-1])
        training_set = training_set.drop(136, 1)
        validation_set = validation_set.drop(136, 1)
        test_set = test_set.drop(136, 1)
        training_set.to_csv("../LTR/MSLR/group_%d/Fold%d/train.txt" % (query_number, i),  sep=' ', header=None, index=False)
        validation_set.to_csv("../LTR/MSLR/group_%d/Fold%d/vali.txt" % (query_number, i),  sep=' ', header=None, index=False)
        test_set.to_csv("../LTR/MSLR/group_%d/Fold%d/test.txt" % (query_number, i),  sep=' ', header=None, index=False)
        print"group_%d Fold%d is over" % (query_number, i)


def main():
    query_number = 40
    group_number = 48
    merge_data(query_number, group_number)


if __name__ == '__main__':
    main()
