# /usr/bin/env python
# coding=utf-8
"""model"""
import torch.nn as nn
from ZEN.modeling import ZenPreTrainedModel, ZenModel


class AvgPoolClassifier(nn.Module):
    """use to get match output"""

    def __init__(self, hidden_size, tag_size, dropout_rate=0.3):
        super(AvgPoolClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, hidden_size))
        self.classifier = nn.Linear(hidden_size, tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        :param input_features: (batch_size, seq_len, hidden_size)
        :return: features_output: (batch_size, 1)
        """
        features_output1 = self.avg_pool(input_features)  # (batch_size, 1, hidden_size)
        features_output1 = nn.ReLU()(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier(features_output1.squeeze(dim=1))  # (batch_size, tag_size)
        return features_output2


class ZenForSequenceClassification(ZenPreTrainedModel):
    """ZEN model for classification.
    This module is composed of the ZEN model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        `input_ngram_ids`: input_ids of ngrams.
        `ngram_token_type_ids`: token_type_ids of ngrams.
        `ngram_attention_mask`: attention_mask of ngrams.
        `ngram_position_matrix`: position matrix of ngrams.

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    """

    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(ZenForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        # pretrain model
        self.bert = ZenModel(config, output_attentions=output_attentions,
                             keep_multihead_output=keep_multihead_output)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                ngram_token_type_ids=None, attention_mask=None, ngram_attention_mask=None,
                output_all_encoded_layers=False, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            input_ngram_ids,
                            ngram_position_matrix,
                            token_type_ids=token_type_ids,
                            ngram_token_type_ids=ngram_token_type_ids,
                            attention_mask=attention_mask,
                            ngram_attention_mask=ngram_attention_mask,
                            output_all_encoded_layers=output_all_encoded_layers,
                            head_mask=head_mask)

        _, pooled_output = outputs

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, tag_size)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
