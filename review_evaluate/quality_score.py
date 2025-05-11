import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def entropy_weight_method(data, is_benefit_list=None):
    """
    熵权法计算权重
    :param data: 原始数据矩阵（DataFrame），每列为一个指标，每行为一个样本
    :param is_benefit_list: 每列是否为正向指标（True 为越大越好，False 为越小越好）
    :return: 各指标的权重（ndarray）
    """
    X = data.copy().astype(float)

    # 1. 标准化
    if is_benefit_list is None:
        is_benefit_list = [True] * X.shape[1]

    for j in range(X.shape[1]):
        col = X.iloc[:, j]
        if is_benefit_list[j]:
            X.iloc[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-12)
        else:
            X.iloc[:, j] = (col.max() - col) / (col.max() - col.min() + 1e-12)

    # 2. 计算比例 pij
    P = X / X.sum(axis=0)

    # 3. 计算熵值 ej
    k = 1 / np.log(X.shape[0])
    E = -k * (P * np.log(P + 1e-12)).sum(axis=0)  # 加上1e-12避免log(0)

    # 4. 计算冗余度 dj
    D = 1 - E

    # 5. 计算权重 wj
    W = D / D.sum()

    return W.values

hedge = []
constructive = []
substan = []
politeness = []
aspect = []
file_list = os.listdir('split_data')
w_dict = {}
for file in tqdm(file_list):
    df = pd.read_csv('split_data/'+file)
    hedge_score = list(df.hedge_score)
    constructive_score = list(df.constructive_score)
    substan_score = list(df.substan_score)
    politeness_score = list(df.politeness_score)
    aspect_score = list(df.aspect_score)
    hedge = hedge + hedge_score
    constructive = constructive + constructive_score
    substan = substan + substan_score
    politeness = politeness + politeness_score
    aspect = aspect + aspect_score
w_dict['Confidence']=hedge
w_dict['Constructive']=constructive
w_dict['Substantiation']=substan
w_dict['Kindness']=politeness
w_dict['Comprehensiveness']=aspect
data = pd.DataFrame(w_dict)
weights = entropy_weight_method(data)

print(round(weights[0],3),round(weights[1],3),round(weights[2],3),round(weights[3],3),round(weights[4],3))
print()
# for file in tqdm(file_list):
#     df = pd.read_csv('split_data/'+file)
#     venue = file.split('.csv')[0]
#     print(file)
#     quality_list = []
#     hedge_score = list(df.hedge_score)
#     constructive_score = list(df.constructive_score)
#     substan_score = list(df.substan_score)
#     politeness_score = list(df.politeness_score)
#     aspect_score = list(df.aspect_score)
#     length = len(hedge_score)
#     for i in tqdm(range(length)):
#         quality_score = hedge_score[i]*weights[0]+constructive_score[i]*weights[1]+substan_score[i]*weights[2]+politeness_score[i]*weights[3]+aspect_score[i]*weights[4]
#         quality_list.append(quality_score)
#     df['quality_score'] = quality_list
#     df.to_csv('quality_result/'+venue+'_quality.csv', index=False)