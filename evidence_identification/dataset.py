import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer



class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        claims = self.data[index]["claim"]
        claim_context = self.data[index]["claim_context"]
        label = self.data[index]["label"]
        # Tokenize the text
        text_encoded = self.tokenizer.encode_plus(
            claims[0], claim_context,
            add_special_tokens=True,  # Add [CLS] at the beginning, [SEP] at the end
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "label": label  # Overall sentence label (e.g., for classification)
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Pad the sequences
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # Pad the labels (token-level span labels)
        label_list = [torch.tensor(instance['label']) for instance in batch]

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "label": torch.tensor(label_list)
        }