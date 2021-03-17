# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: preprocess.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""
import os
import sys

workpath = os.getcwd()
if "_demo" in workpath.replace("\\", "/").split("/")[-1]:
    pass
else:
    os.chdir("../")
    workpath = os.getcwd()
    sys.path.append(workpath)

from collections import Counter

from config.hyper import hyper
from preprocessing.base_class.base_processe_raw_data import BaseProcessing
from preprocessing.converter import DataTypeConverter
from utils.utils import clean_raw_text


class Preprocessing(BaseProcessing):
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.label_set = set()
        self.bio_vocab = {}
        self.vocab_set = Counter()
        self.converter = DataTypeConverter()

    def build_vocab(self):
        """
        根据训练数据构建字典
        :return:
        """
        # 读取训练集数据
        self.one_pass_train()
        # 建立字典
        self.gen_vocab_file()
        self.gen_label_file()

    def one_pass_train(self):
        """
        这里我们默认认为训练数据里面有全部的标签，需要在最初分割数据的时候就要做到
        :return:
        """
        with open(self.train_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label, text = line.split("\t")
                    self.label_set.add(label)
                    self.vocab_set.update(clean_raw_text(text))

    def gen_vocab_file(self, min_freq: int = 0):
        result = ['<pad>', '<unk>']
        for k, v in sorted(self.vocab_set.items(), key=lambda x: x[0]):
            if v > min_freq:
                result.append(k)
        hyper.vocab_num = len(result)
        with open(hyper.vocab_path, 'w', encoding="utf8") as writer:
            for word in result:
                writer.write(word + "\n")

    def gen_label_file(self):
        relation_vocab = ["<unk>"] + list(sorted(self.label_set))
        hyper.label_num = len(relation_vocab)
        with open(hyper.label_path, 'w', encoding="utf8") as writer:
            for label in relation_vocab:
                writer.write(label + "\n")

    def handle_labeled_data(self, in_file_path, out_file_path, mode="same"):
        """
        处理已经有标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """
        return self.converter.raw2process(in_file_path, out_file_path, mode=mode)

    def handle_unlabeled_data(self, in_file_path, out_file_path):
        """
        处理无标注的数据
        :param in_file_path:
        :param out_file_path:
        :return:
        """
        return self.converter.predict2process(in_file_path, out_file_path)


if __name__ == '__main__':
    preprocessor = Preprocessing()
    preprocessor.preprocess()
