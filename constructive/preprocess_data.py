import json
import torch
from torch.utils.data.dataset import random_split
# dataset = pd.read_csv('dataset/toxicbert.csv')
# save_list = []
# for index, row in dataset.iterrows():
#
#     save_dic = {}
#     review_text = row['Text']
#     if row['Target'] == 'N':
#         label = 0
#     else:
#         label = 1
#     save_dic['text'] = review_text
#     save_dic['label'] = label
#     save_list.append(save_dic)
#
# with open('dataset/dataset.json', 'w') as f:
#     json.dump(save_list, f)
a = 0
b = 0
with open('dataset/data.json') as f:
    data = json.load(f)
train_size = int(0.8 * len(data))

test_size = len(data) - train_size
train_data,  test_data = random_split(data, [train_size, test_size])
train_data = list(train_data)
test_data = list(test_data)
with open('dataset/train.json', 'w') as f:
    json.dump(train_data, f)
    
with open('dataset/test.json', 'w') as f:
    json.dump(test_data, f)