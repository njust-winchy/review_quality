import numpy as np
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
# 全局变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_SEED = 400
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SentenceClassifier(nn.Module):
    def __init__(self):
        n_classes = 2
        super(SentenceClassifier, self).__init__()
        self.model = SentenceTransformer('stsb-roberta-base')
        self.bilstm = nn.LSTM(input_size=768,
                              hidden_size=768, batch_first=True, bidirectional=True).to(device)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768 * 2, n_classes).to(device)
        self.sigmoid = nn.Sigmoid()
    def forward(self, sentence):
        input_sentence = torch.from_numpy(self.model.encode(sentence)).to(device)

        last_hidden_out = self.drop(input_sentence)
        output_hidden, _ = self.bilstm(last_hidden_out.view(1, 1, 768))
        output = self.drop(output_hidden)
        output = output.mean(dim=1)

        output = self.out(output)
        return output