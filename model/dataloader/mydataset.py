# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: mydataset.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import json
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config.hyper import hyper
from utils.utils import read_vocab


class MultiHeadDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file_path):
        self.word2index = read_vocab(hyper.vocab_path)
        self.label2index = read_vocab(hyper.label_path)
        self.tokens = []
        self.labels = []
        if hyper.encoder == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(hyper.bert_base_chinese)
        instances = []
        for line in open(file_path, 'r', encoding='utf8'):
            line = line.strip("\n")
            instance = json.loads(line)
            instances.append(instance)
        for instance in instances:
            self.tokens.append(instance['text'])
            self.labels.append(instance['label'])

    def __getitem__(self, index):
        token = self.tokens[index]
        label = self.labels[index]
        if hyper.encoder == "bert":
            tokens_id = self.bert_token2id(token)
        else:
            tokens_id = self.token2tensor(token)
        label_id = self.label2tensor(label)
        return tokens_id, label_id, token, label

    def __len__(self):
        return len(self.tokens)

    def token2tensor(self, text: List[str]) -> torch.tensor:
        text = text[:hyper.max_text_len]
        oov = self.word2index['<unk>']
        padded_list = list(map(lambda x: self.word2index.get(x, oov), text))
        padded_list.extend([self.word2index['<pad>']] * (hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def label2tensor(self, label):
        unk_id = self.label2index['<unk>']
        return self.label2index.get(label, unk_id)

    def bert_token2id(self, text: List[str]):
        text = self.tokenizer.clean_up_tokenization("".join(text))
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        return inputs
