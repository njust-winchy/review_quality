import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

def read_jsonl(file):
    return_list = []
    with open(file, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            return_list.append(data)
    return return_list

nltk.download("punkt")

#tokenizer = AutoTokenizer.from_pretrained('roberta-base')
data_list = read_jsonl('train.jsonl')
process_list = []
count = 0
for line in tqdm(data_list):

    review = line['review']
    labels = line['label']

    for l in labels:
        save_dic = {}
        claim_list = []
        evidence_list = []
        if 'claim' in l[-1]:
            claim = review[l[0]:l[1]]
            # if len(tokenizer.encode_plus(claim)['input_ids']) >max:
            #     max = len(tokenizer.encode_plus(claim)['input_ids'])
            if '\n' in claim:
                claims = claim.split('\n')
                for c in claims:
                    claim_list.append(c)
            else:
                claim_list.append(claim)
            save_dic['review'] = review
            save_dic['claim'] = claim_list
            save_dic['evidence'] = evidence_list

            process_list.append(save_dic)
            count+=1
            continue
        if 'Eval' in l[-1]:
            claim_start = l[0]
            claim_end = l[1]

            claim = review[claim_start:claim_end]
            if '\n' in claim:
                claims = claim.split('\n')
                for c in claims:
                    claim_list.append(c)
            # if len(tokenizer.encode_plus(claim)['input_ids']) >max:
            #     max = len(tokenizer.encode_plus(claim)['input_ids'])
            else:
                claim_list.append(claim)
            for label in labels:
                if label[-1]==l[-1] and label[0]!=l[0]:
                    claim_list.append(review[label[0]:label[1]])

            find_id = l[-1].replace('Eval', 'Jus')
            for m in labels:

                if m[-1] == find_id:
                    evidence_start = m[0]
                    evidence_end = m[1]
                    evidence_list.append(review[evidence_start:evidence_end])
            save_dic['review'] = review
            save_dic['claim'] = claim_list
            save_dic['evidence'] = evidence_list
            if len(process_list)>0:
                if claim_list[0] in process_list[count]['claim']:
                    continue
                else:
                    process_list.append(save_dic)
                    count += 1
            else:
                process_list.append(save_dic)


def extract_context_with_tokens(review, claim, max_tokens=500):
    """
    Extracts a context with full sentences for a given claim from a review,
    ensuring the token count does not exceed max_tokens.

    Parameters:
        review (str): The review text.
        claim (str): The claim to locate in the review.
        max_tokens (int): Maximum allowed tokens for the context.

    Returns:
        str: Extracted context with full sentences within the token limit.
    """
    # Tokenize the review into sentences
    sentences = sent_tokenize(review)

    # Identify the sentence containing the claim
    context_sentences = []
    total_tokens = 0
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        claim_tokens = word_tokenize(claim)
        claim_start_idx = -1
        for i in range(len(tokens) - len(claim_tokens) + 1):
            if tokens[i:i + len(claim_tokens)] == claim_tokens:
                claim_start_idx = i
                break
        if claim_start_idx!=-1:
            # Include the sentence with the claim
            claim_sentence = sentence
            context_sentences.append(claim_sentence)
            total_tokens += len(word_tokenize(claim_sentence))
            break

    if not context_sentences:
        return "Claim not found in the review."

    # Expand context by adding sentences around the claim
    claim_index = sentences.index(claim_sentence)
    # Add preceding sentences
    for i in range(claim_index - 1, -1, -1):
        sentence_tokens = len(word_tokenize(sentences[i]))
        if total_tokens + sentence_tokens <= max_tokens:
            context_sentences.insert(0, sentences[i])
            total_tokens += sentence_tokens
        else:
            break
    # Add succeeding sentences
    for i in range(claim_index + 1, len(sentences)):
        sentence_tokens = len(word_tokenize(sentences[i]))
        if total_tokens + sentence_tokens <= max_tokens:
            context_sentences.append(sentences[i])
            total_tokens += sentence_tokens
        else:
            break

    return " ".join(context_sentences).strip()

def extract_context_paragraph(review, claim, max_tokens_before=100, max_tokens_after=200):
    tokens = word_tokenize(review)

    # 找到 Claim 所在的位置
    claim_tokens = word_tokenize(claim)
    claim_start_idx = -1
    for i in range(len(tokens) - len(claim_tokens) + 1):
        if tokens[i:i + len(claim_tokens)] == claim_tokens:
            claim_start_idx = i
            break

    if claim_start_idx == -1:
        return "Claim not found in review."

    # 根据 Token 索引计算前后上下文
    start_idx = max(0, claim_start_idx - max_tokens_before)
    end_idx = min(len(tokens), claim_start_idx + len(claim_tokens) + max_tokens_after)

    # 拼接上下文并返回
    context_tokens = tokens[start_idx:end_idx]
    context = ' '.join(context_tokens)

    return context

save_list = []
for data in process_list:
    data_dic = {}
    review = data['review']
    split_review = review.split('\n')
    ext_claim = data['claim']
    ext_evidence = data['evidence']
    if len(ext_evidence)==0:
        label = 0
    else:
        label = 1
    claim_context = extract_context_with_tokens(review, ext_claim[0])
    data_dic['claim'] = ext_claim
    data_dic['claim_context'] = claim_context
    data_dic['label'] = label
    save_list.append(data_dic)

with open('evidence_identification/dataset/evi_train.json', 'w') as fp:
    json.dump(save_list, fp)