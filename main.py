# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: main.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import argparse

from model.pl_bert_trainer import BertPlRunner
from model.pl_trainer import PlRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        '-e',
                        type=str,
                        default='bert',
                        help='bert  textcnn')
    parser.add_argument('--mode',
                        '-m',
                        type=str,
                        default='train',
                        help='preprocessing|train|evaluation')
    args = parser.parse_args()
    if args.exp_name == "bert":
        config = BertPlRunner(exp_name=args.exp_name)
    else:
        config = PlRunner(exp_name=args.exp_name)
    config.run(mode=args.mode)
