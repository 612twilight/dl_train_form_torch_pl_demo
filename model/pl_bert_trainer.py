# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: trainer.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam, SGD
from transformers import AdamW

from config.hyper import hyper
from model.dataloader.mydataloader import batch_data_loader
from model.dataloader.mydataset import MyDataset
from model.pl_bert_model import BertPlModel
from preprocessing.preprocess import Preprocessing


class BertPlRunner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = hyper.model_dir

        self.preprocessor = Preprocessing()
        self.optimizer = None
        self.model = None
        self.callbacks = dict()

    def _init_model(self, load_pre=False):
        if not load_pre:
            self.model = BertPlModel()
        else:
            self.model = BertPlModel.load_from_checkpoint(hyper.checkpoint_path)

    def preprocessing(self):
        self.preprocessor.preprocess()

    def _init_callbacks(self):
        checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                              save_top_k=2,
                                              dirpath=hyper.model_dir,
                                              filename='model-{epoch:02d}-{val_f1:.2f}',
                                              mode="max")
        self.callbacks["checkpoint_callback"] = checkpoint_callback

    def _init_trainer(self):
        if hyper.gpus in [0, 1]:
            trainer = pl.Trainer(callbacks=list(self.callbacks.values()), max_epochs=hyper.epoch_num)
        else:
            trainer = pl.Trainer(callbacks=list(self.callbacks.values()), max_epochs=hyper.epoch_num, gpus=hyper.gpus,
                                 accelerator="ddp")
        self.trainer = trainer

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self._init_callbacks()
            self._init_trainer()
            self.train()
        elif mode == 'evaluation':
            self._init_model(load_pre=True, mode=mode)
            self._init_trainer()
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def evaluation(self):
        dev_set = MyDataset(hyper.prepared_dev_file)
        loader = batch_data_loader(dev_set, batch_size=hyper.eval_batch, pin_memory=True)
        self.trainer.test(self.model, test_dataloaders=loader)

    def train(self):
        train_set = MyDataset(hyper.prepared_train_file)
        train_loader = batch_data_loader(train_set,
                                         batch_size=hyper.train_batch,
                                         pin_memory=False, shuffle=True,
                                         num_workers=20)
        dev_set = MyDataset(hyper.prepared_dev_file)
        val_loader = batch_data_loader(dev_set,
                                       batch_size=hyper.eval_batch,
                                       pin_memory=False,
                                       num_workers=20)
        self.trainer.fit(self.model, train_dataloader=train_loader, val_dataloaders=val_loader)
        hyper.checkpoint_path = self.callbacks["checkpoint_callback"].best_model_path
        hyper.save_config()

    def predict(self):
        predict_set = MyDataset(hyper.prepared_predict_file)
        loader = batch_data_loader(predict_set, batch_size=hyper.eval_batch, pin_memory=True)
        all_result = self.trainer.predict(self.model, dataloaders=loader)
        for batch_result in all_result:
            for sample_id, y_pred in batch_result:
                instance = predict_set.instances[sample_id]
                print(instance["label"] + "\t" + "".join(instance["text"]))
                print("predict_result:\t" + predict_set.index2label[y_pred])
                print()
