import torch
import torch.nn as nn
from transformers import RobertaModel


class Constructive_model(nn.Module):
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
