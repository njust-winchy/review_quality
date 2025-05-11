import json
from transformers import AutoTokenizer
from tqdm import tqdm

bert_tokenizer = AutoTokenizer.from_pretrained('roberta-base')


def extract_paragraphs(text):
    paragraphs = []
    start_pos = 0
    length = len(text)
    end_pos = 0
    while start_pos < length:

        while end_pos < length and text[end_pos] not in {'.', '?'}:
            end_pos += 1

        if end_pos < length - 3:
            # Check if the next three characters contain a capital letter
            next_three_chars = text[end_pos + 1:end_pos + 4]
            if any(char.isupper() for char in next_three_chars):
                # Save the paragraph and update start_pos
                paragraphs.append(text[start_pos:end_pos + 1].strip())
                start_pos = end_pos + 1
                end_pos = start_pos
            else:
                # If condition not met, move end_pos to check the next '.'
                end_pos += 1
        else:
            # If end_pos is too close to the end of text
            paragraphs.append(text[start_pos:].strip())
            break

    return paragraphs

id_list = [132,253,287,374,390,413]
save_list = []
claim_list = []
remove_list = ['paper_summary', 'summary_of_strengths', 'summary_of_weaknesses', 'comments,_suggestions_and_typos']
with open('train.jsonl') as f:
    for line in tqdm(f):
        data = json.loads(line)
        review = data['review']
        labels = data['label']
        revw = review.split('\n')
        revw = [r for r in revw if len(r)>15 and r not in remove_list]
        if data['id'] in id_list:
            new_rev = []
            for n in revw:
                if len(bert_tokenizer(n)['input_ids'])>500:
                    span_list = extract_paragraphs(n)
                    revw.remove(n)
                    for x in span_list:
                        new_rev.append(x)
                else:
                    new_rev.append(n)
            for j in new_rev:
                revw.append(j)
        revw = [r for r in revw if len(r) > 15 and r not in remove_list]
        for r in revw:
            sen_dic={}
            label_list = []
            for l in labels:
                if 'Eval' in l[-1] or 'Major' in l[-1]:
                    text = review[l[0]: l[1]]
                    if '\n' in text:
                        claims = text.split('\n')
                        for c in claims:
                            if c in r:
                                label_list.append(c)
                    else:
                        if text in r:
                            label_list.append(text)
            sen_dic['text'] = r
            sen_dic['spans'] = label_list
            claim_list.append(sen_dic)


for c_dic in claim_list:
    save_dic = {}
    if len(c_dic['spans'])==0:
        save_dic['text'] = c_dic['text']
        save_dic['span'] = [[0, 0]]
        save_dic['label'] = 0
        save_list.append(save_dic)
    else:
        span_index = []
        for t in c_dic['spans']:
            start_position = c_dic['text'].find(t)
            end_position = start_position + len(t) - 1 if start_position != -1 else -1
            save_dic['text'] = c_dic['text']
            span_index.append([start_position, end_position])
            save_dic['span'] = span_index
            save_dic['label'] = 1
        save_list.append(save_dic)
with open('claim_train.json', 'w') as fp:
    json.dump(save_list, fp)