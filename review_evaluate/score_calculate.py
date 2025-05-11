import pandas as pd
import json
import ast
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
import os

def calculate_substan_score_new(supported_claims, unsupported_claims, review_length,
                             lambda_value=1.0, alpha=2.0, mu=10.0):
    total_claims = supported_claims + unsupported_claims
    if total_claims == 0:
        return 0.0

    claim_penalty = alpha / (1 + total_claims)
    length_penalty = mu / (1 + review_length)

    denominator = total_claims + lambda_value * unsupported_claims + claim_penalty + length_penalty
    score = supported_claims / denominator

    return round(score, 4)
def preprocess_text(string: str):
    string = string.lower()
    punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~+='''
    string = string.replace('’', " ")
    string = string.replace('\n', " ")
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string

file_list = os.listdir('final_data')
for file in file_list:
    file_pre = file.split('_wo')[0]
    save_file = 'split_data/' + file_pre + '.csv'
    if os.path.exists(save_file):
        print(file+'finished')
        continue

    ploiteness_file = 'politeness_data/' + file_pre + '_politeness_score.json'
    with open(ploiteness_file) as f:
        politeness = json.load(f)
    other_file = 'final_data/' +file
    df = pd.read_csv(other_file)
    Summary_ = df.Summary
    Strengths_ = df.Strengths
    Weaknesses_ = df.Weaknesses
    Questions_ = df.Questions

    review_id_ = df.review_id
    decision_ = df.decision
    summary_claim_ = df.summary_claim
    strengths_claim_ = df.strengths_claim
    weaknesses_claim_ = df.weaknesses_claim
    question_claim_ = df.questions_claim
    aspect_ = df.Aspect
    hedge_ = df.hedge
    constructive_ = df.constructive
    save_data = {}
    save_review_id = []
    save_review = []
    save_decision = []
    save_summary = []
    save_strengths = []
    save_weaknesses = []
    save_questions = []
    save_aspect_score = []
    save_hedge_score = []
    save_constructive_score = []
    save_substan_score = []
    save_politeness_score = []
    save_review_chunk = []
    length = len(review_id_)
    for i in tqdm(range(length)):
        review_id = review_id_[i]
        summary = ast.literal_eval(Summary_[i])
        strengths = ast.literal_eval(Strengths_[i])
        weaknesses = ast.literal_eval(Weaknesses_[i])
        questions = ast.literal_eval(Questions_[i])
        decision = decision_[i]
        review_l = ast.literal_eval(Summary_[i]) + ast.literal_eval(Strengths_[i]) + ast.literal_eval(
            Weaknesses_[i]) + ast.literal_eval(Questions_[i])
        review = ''.join(review_l)
        if len(review)<5:
            continue
        aspect = ast.literal_eval(aspect_[i])
        summary_claim = summary_claim_[i]
        summary_claim_list = ast.literal_eval(summary_claim)
        strengths_claim = strengths_claim_[i]
        strengths_claim_list = ast.literal_eval(strengths_claim)
        weaknesses_claim = weaknesses_claim_[i]
        weaknesses_claim_list = ast.literal_eval(weaknesses_claim)
        question_claim = question_claim_[i]
        question_claim_list = ast.literal_eval(question_claim)
        hedge = hedge_[i]
        hedge_list = ast.literal_eval(hedge)
        supported_claims = 0
        unsupported_claims = 0
        hedge_len = 0
        constructive_len = 0
        constructive = constructive_[i]
        constructive_list = ast.literal_eval(constructive)
        if len(summary_claim_list)>0:
            for summ in summary_claim_list:
                if 'NO EVIDENCE' in summ:
                    unsupported_claims += 1
                else:
                    supported_claims += 1
        if len(strengths_claim_list)>0:
            for stre in strengths_claim_list:
                if 'NO EVIDENCE' in stre:
                    unsupported_claims += 1
                else:
                    supported_claims += 1
        if len(weaknesses_claim_list)>0:
            for weak in weaknesses_claim_list:
                if 'NO EVIDENCE' in weak:
                    unsupported_claims += 1
                else:
                    supported_claims += 1
        if len(question_claim_list)>0:
            for ques in question_claim_list:
                if 'NO EVIDENCE' in ques:
                    unsupported_claims += 1
                else:
                    supported_claims += 1
        if len(hedge_list)>0:
            for hed in hedge_list:
                hedge_len += len(word_tokenize(hed))
        if len(constructive_list) > 0:
            for constru in constructive_list:
                constructive_len += len(word_tokenize(preprocess_text(constru)))
        hedge_score = round(hedge_len / len(word_tokenize(review)), 2)
        constructive_score = round(constructive_len / len(word_tokenize(review)), 2)
        #substan_score = round(calculate_substan_score(supported_claims, unsupported_claims), 2)
        substan_score = calculate_substan_score_new(supported_claims, unsupported_claims, len(word_tokenize(review)))
        politeness_score = round(politeness[review_id], 2)
        # 8 Aspects: Summary, Motivation/Impact,
        # Originality, Soundness/Correctness, Substance,
        # Replicability, Meaningful Comparison, Clarity
        Comprehensiveness = round(len(aspect)/8, 2)
        save_review_id.append(review_id)
        save_review.append(review)
        save_decision.append(decision)
        save_summary.append(summary)
        save_strengths.append(strengths)
        save_weaknesses.append(weaknesses)
        save_questions.append(questions)
        save_hedge_score.append(hedge_score)
        save_constructive_score.append(constructive_score)
        save_substan_score.append(substan_score)
        save_politeness_score.append(politeness_score)
        save_aspect_score.append(Comprehensiveness)

    save_data['review_id'] = save_review_id
    save_data['review'] = save_review
    save_data['decision'] = save_decision
    save_data['summary'] = save_summary
    save_data['strengths'] = save_strengths
    save_data['weaknesses'] = save_weaknesses
    save_data['questions'] = save_questions
    save_data['hedge_score'] = save_hedge_score
    save_data['constructive_score'] = save_constructive_score
    save_data['substan_score'] = save_substan_score
    save_data['politeness_score'] = save_politeness_score
    save_data['aspect_score'] = save_aspect_score
    dc = pd.DataFrame(save_data)

    dc.to_csv(save_file, index=False)