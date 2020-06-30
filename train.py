# -!- coding: utf-8 -!-
"""train with valid"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 参数解析器
import random
import argparse
import os
import logging
from tqdm import trange

import utils
from evaluate import evaluate
from dataloader import NERDataLoader
from model import ZenForSequenceClassification

# 设定参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--epoch_num', required=True, type=int,
                    help="指定epoch_num")
parser.add_argument('--multi_gpu', action='store_true',
                    help="多GPU")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def train(model, data_iterator, optimizer, params):
    """Train the model one epoch
    """
    # set model to training mode
    model.train()

    # 记录平均损失
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    # one epoch
    t = trange(params.train_steps)
    for _ in t:
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, ngram_ids, ngram_positions, \
        ngram_lengths, ngram_seg_ids, ngram_masks = batch

        # compute model output and loss
        loss = model(input_ids,
                     ngram_ids,
                     ngram_positions,
                     token_type_ids=segment_ids,
                     ngram_token_type_ids=ngram_seg_ids,
                     attention_mask=input_mask,
                     ngram_attention_mask=ngram_masks,
                     output_all_encoded_layers=True,
                     labels=label_ids)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        # back-prop
        loss.backward()

        # gradient clipping
        # 梯度截断
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, optimizer, scheduler, params,
                       restore_file=None):
    """Train the model and evaluate every epoch."""
    # load args
    args = parser.parse_args()
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        utils.load_checkpoint(restore_path, model, optimizer)

    # Load training data and val data
    dataloader = NERDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train')
    val_loader = dataloader.get_dataloader(data_sign='val')

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # 一个epoch的步数
        params.train_steps = len(train_loader)

        # Train for one epoch on training set
        train(model, train_loader, optimizer, params)

        # Evaluate for one epoch on training set and validation set
        val_metrics = evaluate(model, val_loader, params, args, mark='Val')  # Dict['loss', 'f1']

        # lr_scheduler学习率递减 step
        scheduler.step()

        # 验证集f1-score
        val_f1 = val_metrics['f1']
        # 提升的f1-score
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.params_path / 'params.json')

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params()
    if not args.multi_gpu:
        # 设置模型使用的gpu
        torch.cuda.set_device(1)
        # 查看现在使用的设备
        print(torch.cuda.current_device())
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.n_gpu = torch.cuda.device_count()

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Set the logger
    utils.set_logger(save=True, log_path=os.path.join(params.params_path, 'train.log'))
    logging.info("Model type: pingan-downstream")
    logging.info("device: {}".format(params.device))

    model = ZenForSequenceClassification.from_pretrained(params.pretrain_model_dir, num_labels=len(params.tags))
    # 保存bert config
    model.to(params.device)
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    # 取模型权重
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_middle = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': params.fin_tuning_lr
         },
        # middle model
        # 衰减
        {'params': [p for n, p in param_middle if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params.weight_decay_rate, 'lr': params.ds_lr
         },
        # 不衰减
        {'params': [p for n, p in param_middle if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': params.ds_lr
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # 学习率递减
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1. / (1. + 0.05 * epoch))

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, optimizer, scheduler, params, args.restore_file)
