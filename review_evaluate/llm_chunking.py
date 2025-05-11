import pandas as pd
from Preprocess_RAG import Find_simtitle
from openai import OpenAI
from tqdm import tqdm
df = pd.read_csv('data/iclr_2019.csv')
paper_title = df.paper_title
review = df.review

prompt_template = ""
with open('Prompt4llmchunking.txt', encoding='utf-8') as f:
    for x in f.readlines():
        prompt_template+=x
f.close()


def deepseek_chunking(prompt):
    client = OpenAI(
        base_url='',
        # required but ignored
        api_key='',
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='gpt-4o',
        temperature=0,
    )
    client.close()
    return chat_completion.choices[0].message.content


def prompt_replace(prompt, exp1, exp2, input):
    prompt = prompt.replace('$Example1', exp1)
    prompt = prompt.replace('$Example2', exp2)
    prompt = prompt.replace('$Input', input)
    return prompt


chunk_list = []
for i in tqdm(range(len(review))):
    rev = review[i]
    title = paper_title[i]
    if i != 0 and title == paper_title[i-1]:
        pass
    else:
        exm_reviews = Find_simtitle(title)
        if len(exm_reviews)<2:
            exm_reviews = Find_simtitle(title)
    rev = rev.replace(title + '.', '')
    example_review1 = exm_reviews[0]
    example_review2 = exm_reviews[1]
    chunking_prompt = prompt_replace(prompt_template, example_review1, example_review2, rev)
    result = deepseek_chunking(chunking_prompt)
    chunk_list.append(result)
df['review_chunking'] = chunk_list
df.to_csv('data/iclr_2019_chunking.csv', index=False)
