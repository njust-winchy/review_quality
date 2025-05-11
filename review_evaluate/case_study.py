import json
from openai import OpenAI
from tqdm import tqdm

prompt_template = ""
with open('case_prompt.txt', encoding='utf-8') as f:
    for x in f.readlines():
        prompt_template+=x
f.close()

def llama_aspect(prompt):
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
    return chat_completion.choices[0].message.content



save_list = []

with open('case_study.json') as f:
    data = json.load(f)
f.close()

for d in tqdm(data):
    prompt_ = prompt_template.replace('$Areview', d['review_1'])
    prompt_input = prompt_.replace('$Breview', d['review_2'])
    output = llama_aspect(prompt_input)
    d['output'] = output
    save_list.append(d)

with open('case_gpt4o_reasoner.json', 'w') as fp:
    json.dump(save_list, fp)
fp.close()