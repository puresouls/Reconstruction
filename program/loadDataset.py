# coding:utf-8
__author__ = 'cheng'

import numpy as np
import pandas as pd


def readDataSet(fileName):
	data = pd.read_csv(fileName)
	dataMat = data.as_matrix(columns=data.columns[1:])
	dataMat = np.nan_to_num(dataMat)
	return dataMat


def handle():
    for i in range(1, 49):
		data_file = "../data/group_40/group_%d.csv" % i
		dataMat = readDataSet(data_file)			 # dataMat带处理的矩阵，一次循环是一组

		filename = "../b-nmf/group_40/%d.txt" % i
		np.savetxt(filename, dataMat, fmt="%d")

if __name__ == "__main__":
	handle()