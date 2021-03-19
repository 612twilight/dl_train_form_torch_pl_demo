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

from model.dataloader.mydataloader import BatchReader


class TextCNNPlModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.filter_sizes = [3, 4, 5]
        self.embedding_dim = 200
        self.num_filter = 64
        self.conv1ds = nn.ModuleList()
        self.max_text_len = 200
        self.label_num = 11
        self.vocab_num = 6201
        for filter_size in self.filter_sizes:
            self.conv1ds.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.embedding_dim,
                        out_channels=self.num_filter,
                        kernel_size=filter_size
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(kernel_size=self.max_text_len - filter_size + 1, stride=1),
                )
            )
        self.embedding = nn.Embedding(num_embeddings=self.vocab_num, embedding_dim=self.embedding_dim)
        self.project_layer = nn.Linear(in_features=len(self.filter_sizes) * self.num_filter,
                                       out_features=self.label_num)
        self.loss_func = CrossEntropyLoss()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        embedding = self.embedding(x)
        x = embedding.permute(0, 2, 1)
        out = [conv(x).unsqueeze(2) for conv in self.conv1ds]
        out = torch.cat(out, dim=2)
        logits = out.view(x.size(0), -1)
        return logits

    def _transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        if isinstance(batch, BatchReader):
            # move all tensors in your custom data structure to the device
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
