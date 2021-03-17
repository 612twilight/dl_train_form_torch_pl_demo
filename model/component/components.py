# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: components.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""



def description(epoch, epoch_num, output):
    return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
        output['loss'].item(), output['crf_loss'].item(),
        output['selection_loss'].item(), epoch, epoch_num)
