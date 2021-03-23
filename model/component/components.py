# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: components.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import torch

from transformers import BatchEncoding


def description(epoch, epoch_num, output):
    return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
        output['loss'].item(), output['crf_loss'].item(),
        output['selection_loss'].item(), epoch, epoch_num)


def transformers_bert_batch(batch_data):
    if not isinstance(batch_data, BatchEncoding):
        return torch.stack(batch_data, 0)
    else:
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for item in batch_data:
            input_ids.append(item["input_ids"].squeeze())
            token_type_ids.append(item["token_type_ids"].squeeze())
            attention_mask.append(item["attention_mask"].squeeze())
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

    pass
