#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""
import pandas as pd
from utils import set_logger
import logging
import random
import math

set_logger()


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, text, tag):
        """
        Desc:
            is_impossible: bool, [True, False]
        """
        self.text = text
        self.tag = tag


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ngram_ids, ngram_positions, ngram_lengths,
                 ngram_tuples, ngram_seg_ids, ngram_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def read_examples(data_dir, data_sign):
    """read BIO-NER data_src to InputExamples
    :return examples (List[InputExample]):
    """
    examples = []
    df = pd.read_csv(data_dir / f'{data_sign}.csv')
    text_list = list(df['sentence'])
    tag_list = list(df['label'])

    for text, tag in zip(text_list, tag_list):
        example = InputExample(text=text.strip().split(' '), tag=int(tag))
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, ngram_dict):
    """convert examples to features.
    :param examples (List[InputExamples]): data_src examples.
    """
    # tag to id

    features = []
    max_len = params.max_seq_length
    for (ex_id, example) in enumerate(examples):
        if ex_id % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_id, len(examples)))

        tokens = [tokenizer.tokenize(c)[0] for c in example.text]
        # cut off
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # mask
        input_mask = [1] * len(input_ids)

        # pad
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # sanity check
        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        # tag
        tag = example.tag

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the word segment from 2 to 7 to check whether there is a word
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the word
                # i is the length of the current word
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        random.shuffle(ngram_matches)
        # max_word_in_seq_proportion = max_word_in_seq
        max_word_in_seq_proportion = math.ceil((len(tokens) / max_len) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]
        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

        import numpy as np
        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_len, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        # Zero-pad up to the max word in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # ----------- code for ngram END-----------

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=tag,
                          ngram_ids=ngram_ids,
                          ngram_positions=ngram_positions_matrix,
                          ngram_lengths=ngram_lengths,
                          ngram_tuples=ngram_tuples,
                          ngram_seg_ids=ngram_seg_ids,
                          ngram_masks=ngram_mask_array))

    return features
