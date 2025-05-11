# prepare_data_parallel.py
import os
import ast
import re
import torch
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize
from transformers import RobertaTokenizerFast
from model.hedge_model import SentenceClassifier
from model.constructive_model import Constructive_model
from model.claim_model import Claim_tagging
from model.evi_model import Evidence_indentify
from tagger.annotator import Annotator

device = torch.device("cuda:0")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Models
claim_model = Claim_tagging('roberta-base', 2).to(device)
claim_model.load_state_dict(torch.load('pretrained_weight/claim_model.pth'), strict=False)
claim_model.eval()

prompt = "Identify subjective evaluation statements about the research that influence the paper's acceptance or rejection decision"
prompt_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

evidence_model = Evidence_indentify('roberta-base', 2).to(device)
evidence_model.load_state_dict(torch.load('pretrained_weight/evidence_model.pth'), strict=False)
evidence_model.eval()

hedge_model = SentenceClassifier().to(device)
hedge_model.load_state_dict(torch.load('pretrained_weight/hedge_model.pt'), strict=False)

constructive_model = Constructive_model('roberta-base', 2).to(device)
constructive_model.load_state_dict(torch.load('pretrained_weight/constructive_model.pth', map_location=device))
constructive_model.eval()

annotator = Annotator('F:\code\substan\\tagger\labels.txt', 'F:\code\confidence score-master\\tagger\seqlab_final', 'gpu')


def text_annotate(texts):
    aspects = []
    for sent in texts:
        if not re.search(r'[0-9a-zA-Z]', sent):
            continue
        asp_sentence = annotator.annotate(sent)
        asp_tags = [tag for _, tag in asp_sentence if tag != 'O'] or ['no_asp']
        aspects.extend(set(asp_tags))
    return aspects

def run_constructive(sentences):
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = constructive_model(encoded['input_ids'], encoded['attention_mask'])
    return torch.argmax(torch.softmax(outputs, dim=1), dim=1).tolist()

def run_hedge(sentences):
    results = []
    for sent in sentences:
        with torch.no_grad():
            pred = hedge_model(sent)
            results.append(torch.argmax(torch.softmax(pred, dim=1)).item())
    return results

def run_claim(sentences):
    batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    batch_size = batch['input_ids'].size(0)
    prompt_batch = prompt_ids.expand(batch_size, -1)
    with torch.no_grad():
        seq_logits, span_logits = claim_model(batch['input_ids'], batch['attention_mask'], prompt_batch)
    seq_pred = torch.argmax(seq_logits, dim=1)
    span_pred = torch.argmax(span_logits, dim=-1)
    return seq_pred.cpu().tolist(), span_pred.cpu()

def run_evidence(claims, context):
    results = []
    for claim in claims:
        inputs = tokenizer(claim, context, padding='max_length', truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            pred = evidence_model(inputs['input_ids'], inputs['attention_mask'])
        label = torch.argmax(pred, dim=1).item()
        results.append(label)
    return results

def process_sentences(sentences, full_text):
    if not sentences:
        return [], [], []

    hedge_out = run_hedge(sentences)
    constructive_out = run_constructive(sentences)
    claim_label, claim_spans = run_claim(sentences)

    claim_texts = []
    for idx, sent in enumerate(sentences):
        if claim_label[idx] == 0:
            continue
        tokens = tokenizer.tokenize(sent)
        pred = claim_spans[idx][:len(tokens)]
        selected = [tok for tok, p in zip(tokens, pred) if p == 1]
        extracted = tokenizer.convert_tokens_to_string(selected)
        if len(extracted) > 10:
            claim_texts.append(extracted)

    evidence_labels = run_evidence(claim_texts, full_text) if claim_texts else []
    evidence_results = []
    for text, label in zip(claim_texts, evidence_labels):
        evidence_results.append(f"{text} {'EVIDENCE' if label else 'NO EVIDENCE'}")
    return hedge_out, constructive_out, evidence_results

def process_section(section_list):
    all_sentences = []
    for s in section_list:
        all_sentences.extend(sent_tokenize(s))
    sentences = [s for s in all_sentences if len(s) > 10 and re.search(r'[0-9a-zA-Z]', s)]
    return sentences

def process_file(filepath):
    df = pd.read_csv(filepath)
    summaries, strengths, weaknesses, questions = df.Summary, df.Strengths, df.Weaknesses, df.Questions

    summary_claim, strengths_claim, weaknesses_claim, questions_claim = [], [], [], []
    hedge_all, constructive_all, aspects_all = [], [], []

    for i in tqdm(range(len(df))):
        sum_list = ast.literal_eval(summaries[i])
        str_list = ast.literal_eval(strengths[i])
        wea_list = ast.literal_eval(weaknesses[i])
        que_list = ast.literal_eval(questions[i])

        asp = text_annotate(sum_list + str_list + wea_list + que_list)
        asp = [a.split('_')[0] for a in asp if a != 'no_asp']
        aspects_all.append(list(set(asp)))

        # Process each section
        hedge, constructive, claims = [], [], []
        for section, claim_list in zip([sum_list, str_list, wea_list, que_list],
                                       [summary_claim, strengths_claim, weaknesses_claim, questions_claim]):
            full_text = ''.join(section)
            sentences = process_section(section)
            h_out, c_out, claims_out = process_sentences(sentences, full_text)

            hedge.extend([s for s, h in zip(sentences, h_out) if h == 1])
            constructive.extend([s for s, c in zip(sentences, c_out) if c == 1])
            claim_list.append(claims_out)

        hedge_all.append(hedge)
        constructive_all.append(constructive)

    df['summary_claim'] = summary_claim
    df['strengths_claim'] = strengths_claim
    df['weaknesses_claim'] = weaknesses_claim
    df['questions_claim'] = questions_claim
    df['hedge'] = hedge_all
    df['constructive'] = constructive_all
    df['Aspect'] = aspects_all

    save_name = os.path.basename(filepath).split('_process')[0]
    df.to_csv(f'final_data/{save_name}_wo_polite.csv', index=False)

if __name__ == '__main__':
    path_list = [os.path.join('output', f) for f in os.listdir('output') if 'iclr_2024' in f]
    for path in path_list:
        process_file(path)
