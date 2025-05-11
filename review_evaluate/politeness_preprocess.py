import pandas as pd
import os
import ast
from tqdm import tqdm
from nltk import sent_tokenize

path_list = os.listdir('output')
for file in path_list:
    if 'process' and 'emnlp' not in file:
        continue
    open_file = 'output/'+file
    df = pd.read_csv(open_file)
    review = list(df.review)
    review_id = list(df.review_id)
    Summary_list = df.Summary
    Strengths_list = df.Strengths
    Weaknesses_list = df.Weaknesses
    Questions_list = df.Questions
    save_id = []
    save_sentence = []
    save_dic = {}
    length = len(Summary_list)

    for i in tqdm(range(length)):
        summary_list = ast.literal_eval(Summary_list[i])
        strengths_list = ast.literal_eval(Strengths_list[i])
        weaknesses_list = ast.literal_eval(Weaknesses_list[i])
        questions_list = ast.literal_eval(Questions_list[i])
        if len(summary_list)!=0:
            for summary in summary_list:
                summ_sentence = sent_tokenize(summary)
                for s in summ_sentence:
                    if len(s)<10:
                        continue
                    save_id.append(review_id[i])
                    save_sentence.append(s)
        if len(strengths_list)!=0:
            for strengths in strengths_list:
                strengths_sentence = sent_tokenize(strengths)
                for s in strengths_sentence:
                    if len(s)<10:
                        continue
                    save_id.append(review_id[i])
                    save_sentence.append(s)
        if len(weaknesses_list)!=0:
            for weaknesses in weaknesses_list:
                weaknesses_sentence = sent_tokenize(weaknesses)
                for s in weaknesses_sentence:
                    if len(s)<10:
                        continue
                    save_id.append(review_id[i])
                    save_sentence.append(s)
        if len(questions_list)!=0:
            for questions in questions_list:
                questions_sentence = sent_tokenize(questions)
                for s in questions_sentence:
                    if len(s)<10:
                        continue
                    save_id.append(review_id[i])
                    save_sentence.append(s)
    save_dic['review_id'] = save_id
    save_dic['review'] = save_sentence
    dc = pd.DataFrame(save_dic)
    file_pre = file.split('_process')[0]
    file_name = 'politeness_data/' + file_pre + '_sentence.csv'
    dc.to_csv(file_name, index=False)
