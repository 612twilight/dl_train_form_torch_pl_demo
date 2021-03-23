# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: mydataloader.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
from functools import partial

import torch
from torch.utils.data.dataloader import DataLoader


class BatchReader(object):
    def __init__(self, data):
        """

        :param data: batch_size个元素
        """
        transposed_data = list(zip(*data))  # 解包的过程，将dataset里的元素重新整合
        self.token_ids = torch.stack(transposed_data[0], 0)
        self.label_ids = torch.tensor(transposed_data[1])
        self.token = transposed_data[2]
        self.label = transposed_data[3]
        self.sample_ids = transposed_data[4]

    def pin_memory(self):
        return self

    def __len__(self):
        return len(self.token)


def collate_fn(batch):
    return BatchReader(batch)


textcnn_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
