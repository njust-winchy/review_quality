import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.data[index]["text"]
        span = self.data[index]["span"]
        label = self.data[index]["label"]
        prompt = "Identify subjective evaluation statements about the research that influence the paper's acceptance or rejection decision"
        # Tokenize the text
        text_encoded = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,  # Add [CLS] at the beginning, [SEP] at the end
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
            padding='max_length',
        )
        prompt_encoded = self.tokenizer.encode_plus(text=prompt, return_tensors="pt", truncation=True, add_special_tokens=True, max_length=512)
        # Get the offsets from the tokenizer
        offsets = text_encoded["offset_mapping"]

        # Create a label sequence for span recognition
        span_labels = [0] * len(offsets)  # Initialize with 0 (non-span)

        # If span is [0, 0], this means there's no span in the sentence
        if span == [[0, 0]]:
            span_labels = [0] * len(offsets)  # No span tokens, all labels are 0
        else:
            # Set the span range to 1
            for s in span:
                start_char, end_char = s  # The span is a list of pairs [start, end]

                start_token = None
                end_token = None

                # Convert the character-level span into token-level span
                for idx, (start_offset, end_offset) in enumerate(offsets):
                    if start_char >= start_offset and start_char < end_offset:
                        start_token = idx
                    if end_char >= start_offset and end_char < end_offset:
                        end_token = idx

                # If no token matches the span, we use the same index for start and end
                if start_token is None:
                    start_token = len(offsets) - 1
                if end_token is None:
                    end_token = start_token

                # Label the span tokens as 1
                for i in range(start_token, end_token + 1):
                    span_labels[i] = 1  # Mark as part of the span

        # Return the tokenized dataset
        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "prompt_ids": prompt_encoded['input_ids'],
            "token_labels": span_labels,  # The token-level span labels
            "label": label  # Overall sentence label (e.g., for classification)
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Pad the sequences
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        prompt_ids_list = [torch.tensor(instance['prompt_ids']) for instance in batch]
        prompt_ids_pad = pad_sequence(prompt_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # Pad the labels (token-level span labels)
        labels_list = [torch.tensor(instance['token_labels']) for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=-1)  # Padding with -1 for labels

        label_list = [torch.tensor(instance['label']) for instance in batch]

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "prompt_ids": torch.tensor(prompt_ids_pad.squeeze(1)),
            "token_labels": torch.tensor(labels_pad),
            "label": torch.tensor(label_list)
        }
