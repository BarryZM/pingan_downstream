# /usr/bin/env python
# coding=utf-8
"""Predict"""

import argparse
import random
import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

import utils
from utils import ID2TAG
from dataloader import NERDataLoader
from model import ZenForSequenceClassification

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--restore_file', required=True,
                    help="Optional, name of the file containing weights to reload before training")


def postprocess(params, preds):
    test_df = pd.read_csv(params.data_dir / 'test.csv')
    id_list = list(test_df['id'])
    with open(params.data_dir / 'submit.csv', 'w', encoding='utf-8') as f:
        for idx, p in zip(id_list, preds):
            f.write(f'{idx}\t{ID2TAG[int(p)]}')


def predict(model, data_iterator, params):
    """Predict entities
    """
    # set model to evaluation mode
    model.eval()

    preds = []
    for batch in tqdm(data_iterator):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, _, input_ngram_ids, ngram_position_matrix, \
        _, ngram_seg_ids, ngram_masks = batch
        with torch.no_grad():
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
    preds = np.concatenate(preds, axis=0)
    preds = np.argmax(preds, axis=1)  # (batch, )
    postprocess(params, preds)


if __name__ == '__main__':
    args = parser.parse_args()
    # 设置模型使用的gpu
    torch.cuda.set_device(7)
    # 查看现在使用的设备
    print('current device:', torch.cuda.current_device())
    # 预测验证集还是测试集
    params = utils.Params()
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # Create the input data pipeline
    logging.info("Loading the dataset...")
    dataloader = NERDataLoader(params)
    test_loader = dataloader.get_dataloader(data_sign='test')
    logging.info("- done.")

    # Define the model
    logging.info('Loading the model...')
    model = ZenForSequenceClassification.from_pretrained(params.pretrain_model_dir, num_labels=len(params.tags))
    model.to(params.device)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'), model)
    logging.info('- done.')

    logging.info("Starting prediction...")
    predict(model, test_loader, params)
    logging.info('- done.')
