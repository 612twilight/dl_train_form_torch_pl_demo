# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: trainer.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import os

import pytorch_lightning as pl
import torch
from prefetch_generator import BackgroundGenerator
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam, SGD
from tqdm import tqdm
from transformers import AdamW

from config.hyper import hyper
from model.component.metrics import F1_triplet, F1_ner
from model.dataloader.mydataloader import textcnn_loader
from model.dataloader.mydataset import MultiHeadDataset
from model.pl_model import TextCNNPlModel
from preprocessing.preprocess import Preprocessing


class PlRunner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = hyper.model_dir

        self.gpu = hyper.gpu
        self.preprocessor = Preprocessing()
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None

    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(model.parameters())
        }
        return m[name]

    def _init_model(self):
        self.model = TextCNNPlModel()

    def preprocessing(self):
        self.preprocessor.preprocess()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=hyper.evaluation_epoch)
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    def evaluation(self):
        dev_set = MultiHeadDataset(hyper.prepared_dev_file)
        loader = textcnn_loader(dev_set, batch_size=hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()
        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))
        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            print('Triplets-> ' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))

    def train(self):
        train_set = MultiHeadDataset(hyper.prepared_train_file)
        train_loader = textcnn_loader(train_set, batch_size=hyper.train_batch, pin_memory=False, shuffle=True)
        dev_set = MultiHeadDataset(hyper.prepared_dev_file)
        val_loader = textcnn_loader(dev_set, batch_size=hyper.eval_batch, pin_memory=False)
        checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                              save_top_k=2,
                                              dirpath=hyper.model_dir,
                                              filename='model-{epoch:02d}-{val_f1:.2f}',
                                              mode="max")
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=hyper.epoch_num)
        trainer.fit(self.model, train_dataloader=train_loader, val_dataloaders=val_loader)
        print(checkpoint_callback.best_model_path)
