# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_pl_demo
File Name: pl_model.py
Author: gaoyw
Create Date: 2021/3/17
-------------------------------------------------
"""
from typing import Any, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoModel,AdamW

from model.dataloader.mydataloader import BatchReader


class BertPlModel(pl.LightningModule):
    def __init__(self):
        # 先于load_from_checkpoint加载，bert_model初始化会被后续check_point覆盖
        super().__init__()
        self.label_num = 11
        self.bert_model = AutoModel.from_pretrained("这里是bert的地址")
        self.project_layer = nn.Linear(in_features=768,
                                       out_features=self.label_num)
        self.loss_func = CrossEntropyLoss()

    def configure_optimizers(self):
        """
        transformers\trainer  create_optimizer_and_scheduler()
        """

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=2e-5)

    def forward(self, x):
        x = self.bert_model(**x)
        logits = self.project_layer(x)
        return logits

    def _transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        if isinstance(batch, BatchReader):
            if isinstance(batch.token_ids, dict):  # 为了 bert 的 tokenizer 输出兼容
                for key in batch.token_ids:
                    batch.token_ids[key] = batch.token_ids[key].to(self.device)
            else:
                batch.token_ids = batch.token_ids.to(device)
            batch.label_ids = batch.label_ids.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device)
        return batch

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        batch = self._transfer_batch_to_device(batch)
        logits = self(batch.token_ids)
        loss = self.loss_func(logits, batch.label_ids)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        batch = self._transfer_batch_to_device(batch)
        logits = self(batch.token_ids)
        loss = self.loss_func(logits, batch.label_ids)
        y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        y_true = batch.label_ids.detach().cpu().numpy()
        return {"loss": loss, "true": y_true, "pred": y_pred}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        y_true = []
        y_pred = []
        for output in outputs:
            y_true.append(output["true"])
            y_pred.append(output["pred"])
        val_f1 = f1_score(y_true=y_true,
                          y_pred=y_pred,
                          average="micro")
        self.log("val_f1", val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        y_true = []
        y_pred = []
        for output in outputs:
            y_true.append(output["true"])
            y_pred.append(output["pred"])
        val_f1 = f1_score(y_true=y_true,
                          y_pred=y_pred,
                          average="micro")
        self.log("val_f1", val_f1)

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        batch = self._transfer_batch_to_device(batch)
        logits = self(batch.token_ids)
        y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        sample_ids = batch.sample_ids
        return zip(sample_ids, y_pred)
