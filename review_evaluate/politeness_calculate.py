import pandas as pd
from nltk import word_tokenize
import json
import os
from tqdm import tqdm
def preprocess_text(string: str):
    string = string.lower()
    punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~+='''
    string = string.replace('’', " ")
    string = string.replace('\n', " ")
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string


def calculate_politeness_score(word_counts):
    
    
    weights = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1}


    weighted_sum = sum(weights[i] * word_counts.get(i, 0) for i in range(1, 6))

    
    total_words = sum(word_counts.values())

 
    return weighted_sum / total_words if total_words > 0 else 0

path_list = os.listdir('score_data')
for file in path_list:
    open_file = 'score_data/' + file
    sp_n = file.split('_politeness')[0]
    df = pd.read_csv(open_file)
    review_id = df.review_id
    reviews = df.reviews
    politeness = df.politeness
    length = len(review_id)
    n = 0
    save_dict = {}
    word_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    id = review_id[0]
    for i in tqdm(range(length)):
        # For 2018
        review_ids = review_id[i]
        if review_ids==id:
            review_sentence = reviews[i]
            word_count = word_tokenize(preprocess_text(review_sentence))
            polite_score = politeness[i]
            word_counts[polite_score] += len(word_count)
        else:
            politeness_score = calculate_politeness_score(word_counts)
            save_dict[id] = politeness_score
            word_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            id = review_ids
            review_sentence = reviews[i]
            word_count = word_tokenize(preprocess_text(review_sentence))
            polite_score = politeness[i]
            word_counts[polite_score] += len(word_count)
    politeness_score = calculate_politeness_score(word_counts)
    save_dict[id] = politeness_score
    file_pre = file.split('_politeness')[0]
    file_name = 'politeness_data/' + file_pre + '_politeness_score.json'
    with open(file_name, 'w') as f:
        json.dump(save_dict, f)