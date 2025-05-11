from tqdm import tqdm
import pandas as pd

data = pd.read_csv('review_data/emnlp_2023_chunking.csv')
#data.drop(58, axis=0, inplace=True)  # axis=0 表示删除行
data = data.drop([941,2247,3034,3602,3684])#2246
save_list = []
Summary_list = []
Strengths_list = []
Weaknesses_list = []
Questions_list = []
for index, row in tqdm(data.iterrows()):
    review_id = row['review_id']
    #review_text = row['review']
    review_text = row['review_chunking']
    summary_index = review_text.find('Summary')
    Strengths_index = review_text.find('Strengths')
    Weaknesses_index = review_text.find('Weaknesses')
    Questions_index = review_text.find('Questions')
    Summary = review_text[summary_index:Strengths_index]
    Strengths = review_text[Strengths_index:Weaknesses_index]
    Weaknesses = review_text[Weaknesses_index:Questions_index]
    Questions = review_text[Questions_index:]
    Summary_split = Summary.split('\n')
    Strengths_split = Strengths.split('\n')
    Weaknesses_split = Weaknesses.split('\n')
    Questions_split = Questions.split('\n')
    Summary_ = []
    Strengths_ = []
    Weaknesses_ = []
    Questions_ = []
    for idx, s in enumerate(Summary_split):
        if len(s)>15 and 'Summary' not in s:
            Summary_.append(s)
    for idx, s in enumerate(Strengths_split):
        if len(s)>15 and 'Strengths' not in s:
            Strengths_.append(s)
    for idx, s in enumerate(Weaknesses_split):
        if len(s)>15 and 'Weaknesses' not in s:
            Weaknesses_.append(s)
    for idx, s in enumerate(Questions_split):
        if len(s)>15 and 'Questions' not in s:
            Questions_.append(s)
    Summary_list.append(Summary_)
    Strengths_list.append(Strengths_)
    Weaknesses_list.append(Weaknesses_)
    Questions_list.append(Questions_)

data['Summary'] = Summary_list
data['Strengths'] = Strengths_list
data['Weaknesses'] = Weaknesses_list
data['Questions'] = Questions_list
data.to_csv('review_data/emnlp_2023_process.csv', index=False)

