# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import logging
from tqdm import tqdm
import torch

import utils
import numpy as np
from sklearn.metrics import f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def evaluate(model, data_iterator, params, args, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    preds = []
    out_label_ids = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    for batch in tqdm(data_iterator):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_ngram_ids, ngram_position_matrix, \
        ngram_lengths, ngram_seg_ids, ngram_masks = batch

        with torch.no_grad():
            # get loss
            loss = model(input_ids,
                         input_ngram_ids,
                         ngram_position_matrix,
                         token_type_ids=segment_ids,
                         ngram_token_type_ids=ngram_seg_ids,
                         attention_mask=input_mask,
                         ngram_attention_mask=ngram_masks,
                         output_all_encoded_layers=True,
                         labels=label_ids)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()
            loss_avg.update(loss.item())

            logits = model(input_ids=input_ids,
                           input_ngram_ids=input_ngram_ids,
                           ngram_position_matrix=ngram_position_matrix,
                           token_type_ids=segment_ids,
                           ngram_token_type_ids=ngram_seg_ids,
                           attention_mask=input_mask,
                           ngram_attention_mask=ngram_masks,
                           output_all_encoded_layers=True,
                           )

        preds.append(logits.detach().cpu().numpy())
        out_label_ids.append(label_ids.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    preds = np.argmax(preds, axis=1)  # (batch, )
    out_label_ids = np.concatenate(out_label_ids, axis=0)
    metrics = get_metrics(preds, out_label_ids)

    # logging loss, f1 and report
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    return metrics
