from selenium.webdriver.common.by import By
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import json
import chromedriver_autoinstaller
from tqdm import tqdm
from selenium import webdriver
file_name = os.listdir("../ICLR_2024_url") # Taking ICLR 2024 as an example
paper_title_list = []
paper_url_list = []
for js in file_name:
    with open('ICLR_2024_url\\'+js, encoding='utf-8') as f:
        dic = json.load(f)
        for p in tqdm(dic):
            url = p['url']
            paper_title = p['title']
            paper_title_list.append(paper_title)
            paper_url_list.append(url)
rag_dic = dict(zip(paper_title_list, paper_url_list))
rag_dic_keys = list(rag_dic.keys())
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  

title_embeddings = model.encode(rag_dic_keys)


def Find_simtitle(compare_title):
  

    query_embedding = model.encode(compare_title)

    
    similarities = util.cos_sim(query_embedding, title_embeddings)

   
    most_similar_index = np.argmax(similarities)
    most_similar_title = rag_dic_keys[most_similar_index]
    url = rag_dic[most_similar_title]

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chromedriver_autoinstaller.install()
    browser = webdriver.Chrome(options=chrome_options)
    browser.get(url)
    time.sleep(4)
    k = browser.find_elements(By.CLASS_NAME, "note-content-field")
    title = []
    for y in range(len(k)):
        if 'Summary:' in k[y].text:
            title.append(y)
        elif 'Strengths:' in k[y].text:
            title.append(y)
        elif 'Weaknesses:' in k[y].text:
            title.append(y)
        elif 'Questions:' in k[y].text:
            title.append(y)
        elif 'Soundness:' in k[y].text:
            title.append(y)
        elif 'Presentation:' in k[y].text:
            title.append(y)
        elif 'Contribution:' in k[y].text:
            title.append(y)
        elif 'Rating:' in k[y].text:
            title.append(y)
        elif 'Confidence:' in k[y].text:
            title.append(y)

    v = browser.find_elements(By.CLASS_NAME, "note-content-value")
    reviews = []
    for i in range(len(title) // 9):
        #review = {}
        m = i * 9
        Summary = v[title[m]].text
        Strengths = v[title[m + 4]].text
        Weakness = v[title[m + 5]].text
        Questions = v[title[m + 6]].text
        #review['Summary'] = Summary
        #review['Strengths'] = Strengths
        #review['Weaknesses'] = Weakness
        #review['Questions'] = Questions
        review = "Summary:\n" + Summary + "\nStrengths:\n" + Strengths + "\nWeaknesses:\n" + Weakness + "\nQuestions:\n" + Questions
        reviews.append(review)
    return reviews[0:2]
