import torch
import torch.nn as nn
from transformers import RobertaModel
import torch.nn.functional as F


class MyRobertaModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name_or_path)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.encoder.config.hidden_size,
                                                            nhead=8, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)

        # Fully connected layers

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification
        # Activation and dropout
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        #classification_input = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        transformer_output = self.transformer_encoder(sequence_output)
        #att_output, _ = self.attention(claim_sequence_output, context_sequence_output, context_sequence_output)
        classification_input = torch.max(transformer_output, dim=1)[0]
        # Classification layers
        x = self.relu(self.fc1(classification_input))
        logits = self.fc2(x)  # Shape: (batch_size, 1)

        return logits

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