import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import os
comp_dic = {}
file_list = os.listdir('quality_result')
save_dic = {}
venue_list = []
quality_list = []
year_list = []
for file in tqdm(file_list):
    venue = file.split('_quality')[0]
    df = pd.read_csv('quality_result/'+file)
    quality_score = df.quality_score
    if 'acl' in file:
        year = 2017
    elif 'arr' in file:
        year = 2022
    elif 'coling' in file:
        year = 2020
    elif 'conll' in file:
        year = 2016
    else:
        fr = file.split('_quality')[0]
        year = int(fr.split('_')[1])
    for i in quality_score:
        quality_list.append(i)
        venue_list.append(venue)
        year_list.append(year)
    quality = list(df.quality_score)
    comp_dic[venue]=sum(quality)/len(quality)
save_dic['weighted_score']=quality_list
save_dic['conference']=venue_list
save_dic['year']=year_list
dc = pd.DataFrame(save_dic)
mean_score = dc['weighted_score'].mean()
std_score = dc['weighted_score'].std()
plt.figure(figsize=(12, 6))
sns.histplot(dc['weighted_score'], kde=True, bins=30, color='skyblue', edgecolor='black')
plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean = {mean_score:.2f}')
plt.axvline(mean_score + std_score, color='green', linestyle='--', label=f'Mean + 1σ = {mean_score + std_score:.2f}')
plt.axvline(mean_score - std_score, color='green', linestyle='--', label=f'Mean - 1σ = {mean_score - std_score:.2f}')

plt.title("Average review quality score distribution", fontsize=16, fontweight='bold')
plt.xlabel("Quality score", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("output/distribution.png", dpi=400, bbox_inches='tight')  # 保存图像
plt.show()
print()

plt.figure(figsize=(12, 6), dpi=400)
sns.boxplot(data=dc, x='conference', y='weighted_score', palette='Set2')
plt.title("Average review quality score distribution by conference", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel("Quality score", fontsize=14)
plt.xlabel("Conference", fontsize=14)

plt.tight_layout()
plt.savefig("output/conference.png", dpi=400, bbox_inches='tight')
plt.show()



yearly_avg = dc.groupby('year')['weighted_score'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=yearly_avg, x='year', y='weighted_score', palette='coolwarm')
plt.title("Average review scores by year", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average quality score", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("output/year.png", dpi=400, bbox_inches='tight')
plt.show()