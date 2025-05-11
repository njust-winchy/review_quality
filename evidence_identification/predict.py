import torch
from model import MyRobertaModel
from transformers import RobertaTokenizerFast
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda:0")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')  # AutoTokenizer

model = MyRobertaModel('roberta-base', 2)

model.load_state_dict(torch.load('weights/roberta-base-Dec28_22-24-20-epoch11-macro_f10.847.pth'), strict=False)
claim = 'The training objective is reasonable. In particular, high-level features show translation invariance. '
context = "Identify subjective evaluation statements about the research that influence the paper's acceptance or rejection decision"

text_encoded = tokenizer.encode_plus(
            claim, context,
            add_special_tokens=True,  # Add [CLS] at the beginning, [SEP] at the end
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
            padding='max_length',
        )
input_ids_list = [torch.tensor(text_encoded['input_ids'])]
input_ids_pad = torch.tensor(pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id))
attention_mask_list = [torch.tensor(text_encoded['attention_mask'])]
attention_mask_pad = torch.tensor(pad_sequence(attention_mask_list, batch_first=True, padding_value=0))
seq_logits = model(input_ids_pad, attention_mask_pad)
pred_label = torch.max(seq_logits, dim=1)[1]
