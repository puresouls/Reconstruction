# coding:utf-8
__author__ = 'cheng'

# 查询分组法


import os
import numpy as np
import pandas as pd


def query_group(query_number):
    data_part = pd.read_csv("../data/data_part_beyond_20.csv")  # 243093个doc， 1911个qid
    data_part['136'] = map(lambda x: x.split(':')[-1], data_part['136'])
    data_part['136'] = data_part['136'].astype(int)
    feats_list = ['0', '1', '135', '136', '10000']
    data_part = data_part[feats_list]

    groups = data_part.groupby(['1'])

    group_path = "../data/group_%d" % query_number
    if not os.path.isdir(group_path):
        os.makedirs(r'group_%d' % query_number)

    group_num = 0
    qid_num = 0
    df = pd.DataFrame(columns=[])
    for name, value in groups:
        qid_num += 1
        if qid_num % query_number == 0 or qid_num == 1911:
            df = df.append(value.iloc[:], ignore_index=True)
            df = pd.pivot_table(df, index='1', columns='10000', values='136', aggfunc=np.mean)
            group_num += 1
            group_num_path = "../data/group_%d/group_%d.csv" % (query_number, group_num)
            df.to_csv(group_num_path)
            print "%d finished" % group_num
            df = pd.DataFrame(columns=[])
        else:
            df = df.append(value.iloc[:], ignore_index=True)


def readDataSet(fileName):
    data = pd.read_csv(fileName)
    dataMat = data.as_matrix(columns=data.columns[1:])
    dataMat = np.nan_to_num(dataMat)
    return dataMat


def main():
    num = [5, 10, 20, 30, 40]
    for query_num in num:
        query_group(query_num)


if __name__ == '__main__':
    main()