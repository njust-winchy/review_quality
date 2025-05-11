import os
import sys
import torch


num_classes = 2
span_classes = 512
epochs = 20
batch_size = 4
lr = 5e-4
weight_decay = 1e-4
freeze_layers = True

pretrained_model_name_or_path = 'roberta-base'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
weights_name = 'claim_model.pth'
data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
save_weights_path = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录

