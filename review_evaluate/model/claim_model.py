import torch
import torch.nn as nn
from transformers import RobertaModel
import torch.nn.functional as F


class Claim_tagging(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name_or_path)
        self.claim_classification = claim_Classification(num_classes)
        # for param in self.roberta.parameters():
        #      param.requires_grad = False
        self.attention = AttentionLayer(768, attention_type='additive')
        self.span_classifier = nn.Sequential(
            nn.Linear(768, 128),  # 768 -> 128
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 2)  # 128 -> 2
        )

    def forward(self, input_ids, attention_mask, prompt):
        # Sequence classification
        out_backbone = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out_backbone['last_hidden_state']
        cls_rep = last_hidden_state[:, 0, :]

        classification_res = self.claim_classification(cls_rep)

        prompt_backbone = self.roberta(input_ids=prompt)
        prompt_out = prompt_backbone['last_hidden_state']
        # Token classification
        output, att_weight = self.attention(prompt_out, last_hidden_state, last_hidden_state) # b,s,d_model
        span_logits = self.span_classifier(output) #4,512,2

        return classification_res, span_logits



class claim_Classification(nn.Module):
    def __init__(self, num_classes):
        super(claim_Classification, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.classification = nn.Linear(128, num_classes)


    def forward(self, sequence):
        projected_output = self.projection(sequence)
        lstm_out, _ = self.lstm(projected_output)

        out_fc2 = self.classification(projected_output)
        return out_fc2



class AttentionLayer(nn.Module):
    def __init__(self, d_model, attention_type="scaled_dot_product"):
        """
        自定义注意力层
        :param d_model: 输入特征的维度
        :param attention_type: 注意力类型，默认为 scaled dot-product
        """
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.attention_type = attention_type

        # 可学习参数（对于可选的加性注意力）
        if attention_type == "additive":
            self.W_query = nn.Linear(d_model, d_model)
            self.W_key = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, 1)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        :param query: [batch_size, query_len, d_model]
        :param key: [batch_size, seq_len, d_model]
        :param value: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len]，可选
        :return: 注意力输出和权重
        """
        if self.attention_type == "scaled_dot_product":
            # 1. 计算点积注意力
            scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, query_len, seq_len]
            scores = scores / (self.d_model ** 0.5)  # 缩放

        elif self.attention_type == "additive":
            # 1. 计算加性注意力
            q_proj = self.W_query(query).unsqueeze(2)  # [batch_size, query_len, 1, d_model]
            k_proj = self.W_key(key).unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
            scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [batch_size, query_len, seq_len]

        # 2. 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 3. 归一化权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, query_len, seq_len]

        # 4. 加权求和
        output = torch.bmm(attention_weights, value)  # [batch_size, query_len, d_model]
        output = torch.bmm(attention_weights.transpose(1, 2), output)
        return output, attention_weights