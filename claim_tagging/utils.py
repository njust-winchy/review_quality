import sys
import json
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def train_one_epoch(model, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)
    predicted_span_labels = torch.LongTensor([]).to(device)
    ground_truth_span_labels = torch.LongTensor([]).to(device)
    loss_function = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()
    count = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        prompt = data['prompt_ids'].to(device)
        token_label = data['token_labels'].to(device)
        label = data['label'].to(device)
        seq_logits, span_logits = model(input_ids=input_ids, attention_mask=attention_mask, prompt=prompt)
        logits_flat = span_logits.view(-1, span_logits.size(-1))  # (B*T, C)
        labels_flat = token_label.view(-1)  # (B*T)
        attention_mask_flat = attention_mask.view(-1)  # (B*T)
        pred_label = torch.max(seq_logits, dim=1)[1]
        ground_truth_labels = torch.cat([ground_truth_labels, label])
        predicted_labels = torch.cat([predicted_labels, pred_label])
        # Apply the attention mask
        valid_indices = attention_mask_flat == 1
        logits_flat = logits_flat[valid_indices]
        labels_flat = labels_flat[valid_indices]

        seq_loss = loss_function(seq_logits, label)
        span_loss = loss_function(logits_flat, labels_flat)

        predictions = torch.argmax(span_logits, dim=-1)  # Shape: (B, T)
        valid_tokens = attention_mask.view(-1) == 1  # Flattened mask

        # Flatten predictions and labels
        predictions_flat = predictions.view(-1)
        labels_flat = token_label.view(-1)

        # Apply mask
        predictions_flat = predictions_flat[valid_tokens]
        labels_flat = labels_flat[valid_tokens]


        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        #accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        #predictions_np = predictions_flat.cpu().numpy()
        #labels_np = labels_flat.cpu().numpy()
        ground_truth_span_labels = torch.cat([ground_truth_span_labels, labels_flat])
        predicted_span_labels = torch.cat([predicted_span_labels, predictions_flat])
        # Calculate F1 Score (macro, weighted, etc.)
        accuracy = accuracy_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        weighted_recall = recall_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(), average='weighted', zero_division=0)
        weighted_f1 = f1_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(), average='weighted', zero_division=0)
        weighted_prec = precision_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(), average='weighted', zero_division=0)
        classification_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted', zero_division=0)
        loss = 0.6*seq_loss + span_loss
        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, class_f1: {:.3f}, span_acc: {:.3f}, weighted_recall: {:.3f}, weighted_f1: {:.3f}, weighted_prec: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss, classification_f1,
            accuracy, weighted_recall, weighted_f1, weighted_prec
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        #if count%4==0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        # else:
        #     count+=1


    return {
        'loss': avg_loss,
        'classification_f1': classification_f1,
        'span_accuracy': accuracy,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_prec': weighted_prec
    }


@torch.no_grad()
def validate(model, device, data_loader, epoch=0):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)
    predicted_span_labels = torch.LongTensor([]).to(device)
    ground_truth_span_labels = torch.LongTensor([]).to(device)
    loss_function = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)  # 累计损失
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        prompt = data['prompt_ids'].to(device)
        token_label = data['token_labels'].to(device)
        label = data['label'].to(device)
        seq_logits, span_logits = model(input_ids=input_ids, attention_mask=attention_mask, prompt=prompt)
        logits_flat = span_logits.view(-1, span_logits.size(-1))  # (B*T, C)
        labels_flat = token_label.view(-1)  # (B*T)
        attention_mask_flat = attention_mask.view(-1)  # (B*T)
        pred_label = torch.max(seq_logits, dim=1)[1]
        ground_truth_labels = torch.cat([ground_truth_labels, label])
        predicted_labels = torch.cat([predicted_labels, pred_label])
        # Apply the attention mask
        valid_indices = attention_mask_flat == 1
        logits_flat = logits_flat[valid_indices]
        labels_flat = labels_flat[valid_indices]

        seq_loss = loss_function(seq_logits, label)
        span_loss = loss_function(logits_flat, labels_flat)

        predictions = torch.argmax(span_logits, dim=-1)  # Shape: (B, T)
        valid_tokens = attention_mask.view(-1) == 1  # Flattened mask

        # Flatten predictions and labels
        predictions_flat = predictions.view(-1)
        labels_flat = token_label.view(-1)

        # Apply mask
        predictions_flat = predictions_flat[valid_tokens]
        labels_flat = labels_flat[valid_tokens]

        ground_truth_span_labels = torch.cat([ground_truth_span_labels, labels_flat])
        predicted_span_labels = torch.cat([predicted_span_labels, predictions_flat])
        # Calculate F1 Score (macro, weighted, etc.)
        accuracy = accuracy_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        weighted_recall = recall_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(),
                                       average='weighted', zero_division=0)
        weighted_f1 = f1_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(), average='weighted',
                               zero_division=0)
        weighted_prec = precision_score(ground_truth_span_labels.tolist(), predicted_span_labels.tolist(),
                                        average='weighted', zero_division=0)
        classification_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted',
                                     zero_division=0)
        loss = seq_loss + span_loss

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, class_f1: {:.3f}, span_acc: {:.3f}, weighted_recall: {:.3f}, weighted_f1: {:.3f}, weighted_prec: {:.3f}".format(
            epoch, avg_loss, classification_f1,
            accuracy, weighted_recall, weighted_f1, weighted_prec
        )

    return {
        'loss': avg_loss,
        'classification_f1': classification_f1,
        'span_accuracy': accuracy,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_prec': weighted_prec
    }