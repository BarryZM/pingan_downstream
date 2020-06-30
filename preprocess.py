# /usr/bin/env python
# coding=utf-8
"""preprocess"""
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from utils import MAP_DICT

# 最大文本长度max len
SPLIT_LEN = 256
DATA_DIR = Path('./data')


def truncate_text(text_list):
    """
    cut off
    """
    idx = 0
    while idx < len(text_list):
        text = text_list[idx].strip().split(' ')
        if len(text) > SPLIT_LEN:
            # find new start
            start = len(text) - SPLIT_LEN
            new_text = text[start:]
            text_list[idx] = " ".join(new_text)
        idx += 1
    return text_list


def generate_data():
    # train_set
    train_data = pd.read_excel(DATA_DIR / 'train.xlsx')
    id_list = train_data['id'].tolist()
    id_set = set(id_list)
    # init
    train_sentence_list = []
    train_label_list = []
    for t_id in tqdm(id_set, desc='train set'):
        df_idx = train_data[train_data['id'] == t_id].copy()
        category_list = df_idx['category'].tolist()
        char_list = df_idx['char'].tolist()
        label_list = df_idx['label'].tolist()
        sentence = ''
        for index, cate in enumerate(category_list):
            # 人 预测人的意图，构造训练数据必然是人的回答作为最后一句话，标签也是人的最后一句话作为标签
            if cate == 0:
                sentence += " ".join(
                    char_list[index].replace("'", "").replace("[", "").replace("]", "").split(', ')) + " 人 "
                train_sentence_list.append(sentence)
                train_label_list.append(label_list[index])
            # 机器人的话语作为起点
            else:
                sentence = " ".join(
                    char_list[index].replace("'", "").replace("[", "").replace("]", "").split(', ')) + " 机 "

    train_sentence_list = truncate_text(train_sentence_list)

    train_dict = {'sentence': train_sentence_list, 'label': train_label_list}
    train_df = pd.DataFrame(train_dict)
    train_df['label'] = train_df['label'].apply(lambda row: MAP_DICT[row])
    train_df.to_csv(DATA_DIR / 'train_src.csv', encoding='utf-8', index=False)

    # test_set
    test_data = pd.read_excel(DATA_DIR / 'public_test.xlsx')
    # test_id顺序不能乱
    test_id_list = test_data['id'].tolist()
    order_test_id = []
    for t_id in test_id_list:
        if t_id not in order_test_id:
            order_test_id.append(t_id)

    test_sentence_list = []
    test_label_list = []
    test_id_list = []
    for t_id in tqdm(order_test_id, desc='test set'):
        id_pd = test_data[test_data['id'] == t_id].copy()
        category_list = id_pd['catgory'].tolist()
        char_list = id_pd['char'].tolist()
        sentence = ''
        for index, cate in enumerate(category_list):
            if cate == 0:
                sentence += " ".join(
                    char_list[index].replace("'", "").replace("[", "").replace("]", "").split(', ')) + " 人 "
                test_sentence_list.append(sentence)
                test_label_list.append(-1)
                test_id_list.append(t_id)
            else:
                sentence = " ".join(
                    char_list[index].replace("'", "").replace("[", "").replace("]", "").split(', ')) + " 机 "

    truncate_text(test_sentence_list)

    test_dict = {'id': test_id_list, 'sentence': test_sentence_list, 'label': test_label_list}
    test_pd = pd.DataFrame(test_dict)
    test_pd.to_csv(DATA_DIR/'test.csv', encoding='utf8', index=False)

    # split
    total_train = pd.read_csv(DATA_DIR / 'train_src.csv', encoding='utf-8')
    train_data = total_train[:-8000]
    val_data = total_train[-8000:]
    train_data.to_csv(DATA_DIR / 'train.csv', encoding='utf-8', index=False)
    val_data.to_csv(DATA_DIR / 'val.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    generate_data()
