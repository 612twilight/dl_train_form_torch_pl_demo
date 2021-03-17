# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: converter.py
Author: gaoyw
Create Date: 2020/12/4
-------------------------------------------------
"""
import hashlib
import json

from tqdm import tqdm

from preprocessing.base_class.base_converter import BaseDataTypeConverter
from utils.utils import clean_raw_text


class DataTypeConverter(BaseDataTypeConverter):
    def __init__(self):
        super(DataTypeConverter, self).__init__()
        self.mid_data = []

    def raw2mid_data(self, raw_path):
        with open(raw_path, 'r', encoding='utf8') as s:
            lines = s.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line:
                    label, text = line.split("\t")
                    clean_text = clean_raw_text(text)
                    self.mid_data.append(
                        {"sample_id": hashlib.md5(clean_text.encode()).hexdigest(), "label": label, "raw_text": text,
                         "text": [word for word in clean_text]})

    def mid_data2process(self, process_path):
        with open(process_path, 'w', encoding='utf8') as t:
            for sample in self.mid_data:
                t.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def predict2mid_data(self, predict_path):
        """
        没有标签的数据，但是对于这个问题，需要填上空标签
        :param predict_path:
        :return:
        """
        with open(predict_path, 'r', encoding='utf8') as s:
            for line in s:
                line = line.strip()
                if line:
                    label, text = line.split("\t")
                    clean_text = clean_raw_text(text)
                    self.mid_data.append(
                        {"sample_id": hashlib.md5(clean_text.encode()).hexdigest(), "label": label, "raw_text": text,
                         "text": [word for word in text]})

    def mid_data_balance(self):
        pass
