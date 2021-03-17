# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: utils.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""

import re

plain_pattern = re.compile("\\s+")


def clean_raw_text(raw_text):
    clean_text = plain_pattern.sub("", raw_text)
    return clean_text


def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
    vocab = [line.strip() for line in lines]
    return dict(zip(vocab, range(len(vocab))))
