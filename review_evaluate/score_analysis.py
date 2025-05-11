import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
import os
file_list = os.listdir('split_data')

# 每列对应一个评分指标
dfs = [pd.read_csv('split_data/' + f) for f in file_list]
merged_df = pd.concat(dfs, ignore_index=True)

# 需要分析的评分列
score_columns = ["aspect_score", "constructive_score", "substan_score", "hedge_score", "politeness_score"]

# 计算离散度统计量
stats = pd.DataFrame({
    "Mean": merged_df[score_columns].mean(),
    "Std": merged_df[score_columns].std(),
    "Variance": merged_df[score_columns].var(),
    "CV": merged_df[score_columns].std() / merged_df[score_columns].mean()
}).round(3)

print("离散度统计表：")
print(stats)

