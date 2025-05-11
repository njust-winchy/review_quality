import torch
from model import MyRobertaModel
from transformers import RobertaTokenizerFast
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda:0")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')  # AutoTokenizer

model = MyRobertaModel('roberta-base', 2)

model.load_state_dict(torch.load('weights/roberta-base-Dec05_01-04-34-epoch13-macro_f10.892.pth'), strict=False)
sentence = 'The training objective is reasonable. In particular, high-level features show translation invariance. '
prompt = "Identify subjective evaluation statements about the research that influence the paper's acceptance or rejection decision"

text_encode = tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,  # Add [CLS] at the beginning, [SEP] at the end
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
            padding='max_length',
        )
prompt_encoded = tokenizer.encode_plus(text=prompt, return_tensors="pt", truncation=True, add_special_tokens=True,
                                            max_length=512)
input_ids_list = [torch.tensor(text_encode['input_ids'])]
input_ids_pad = torch.tensor(pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id))
prompt_ids_list = [torch.tensor(prompt_encoded['input_ids'])]
prompt_ids_pad = torch.tensor(pad_sequence(prompt_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id))
attention_mask_list = [torch.tensor(text_encode['attention_mask'])]
attention_mask_pad = torch.tensor(pad_sequence(attention_mask_list, batch_first=True, padding_value=0))
seq_logits, span_logits = model(input_ids_pad, attention_mask_pad, prompt_ids_pad.squeeze(1))

logits_flat = span_logits.view(-1, span_logits.size(-1))  # (B*T, C)
attention_mask_flat = attention_mask_pad.view(-1)
valid_indices = attention_mask_flat == 1
logits_flat = logits_flat[valid_indices]
pred_ = torch.max(logits_flat, dim=1)[1]
pred_label = torch.max(seq_logits, dim=1)[1]
tokens = tokenizer.tokenize(sentence)
selected_tokens = [token for token, pred in zip(tokens, pred_) if pred == 1]
extracted_text = tokenizer.convert_tokens_to_string(selected_tokens)

print()