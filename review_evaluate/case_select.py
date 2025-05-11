import pandas as pd
import os
from tqdm import tqdm
import json
file_list = os.listdir('quality_result')
dfs = [pd.read_csv('quality_result/' + f) for f in file_list]
merged_df = pd.concat(dfs, ignore_index=True)
Sorted = merged_df.sort_values(['quality_score'], ascending=False)
review = list(Sorted.review)
review_id = list(Sorted.review_id)
hedge_score = list(Sorted.hedge_score)
constructive_score = list(Sorted.constructive_score)
substan_score = list(Sorted.substan_score)
politeness_score = list(Sorted.politeness_score)
aspect_score = list(Sorted.aspect_score)
quality_score = list(Sorted.quality_score)
length = len(quality_score)
save_list = []
for i in range(length):
    save_dic = {}
    if i == 5:
        break
    review_text = review[i]
    review_id_1 = review_id[i]
    score_const = constructive_score[i]
    score_substan = substan_score[i]
    score_asp = aspect_score[i]
    score_polite = politeness_score[i]
    score_hedge = hedge_score[i]
    score_quality = quality_score[i]
    comp_review_text = review[-10000-i]
    comp_review_id = review_id[-10000 - i]
    comp_score_const = constructive_score[-10000-i]
    comp_score_substan = substan_score[-10000-i]
    comp_score_asp = aspect_score[-10000-i]
    comp_score_polite = politeness_score[-10000-i]
    comp_score_hedge = hedge_score[-10000-i]
    comp_score_quality = quality_score[-10000-i]
    save_dic['review_1'] = review_text
    save_dic['review_id_1'] = review_id_1
    save_dic['score_const_1'] = score_const
    save_dic['score_substan_1'] = score_substan
    save_dic['score_asp_1'] = score_asp
    save_dic['score_polite_1'] = score_polite
    save_dic['score_hedge_1'] = score_hedge
    save_dic['score_quality_1'] = score_quality

    save_dic['review_2'] = comp_review_text
    save_dic['review_id_2'] = comp_review_id
    save_dic['score_const_2'] = comp_score_const
    save_dic['score_substan_2'] = comp_score_substan
    save_dic['score_asp_2'] = comp_score_asp
    save_dic['score_polite_2'] = comp_score_polite
    save_dic['score_hedge_2'] = comp_score_hedge
    save_dic['score_quality_2'] = comp_score_quality
    save_list.append(save_dic)

with open('case_study.json', 'w') as f:
    json.dump(save_list, f)
print()
# for file in file_list:
#     venue = file.split('_quality')[0]
#     df = pd.read_csv('quality_result/'+file)
#     decision = df.decision
#     com = {0:0,1:0}
#     for d in decision:
#         if 'Accept' in d:
#             com[1]+=1
#         else:
#             com[0]+=1
#     accept_rate = com[1]/(com[0]+com[1])
#     print(venue+':'+str(accept_rate))