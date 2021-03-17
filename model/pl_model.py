# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_pl_demo
File Name: pl_model.py
Author: gaoyw
Create Date: 2021/3/17
-------------------------------------------------
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


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

    def training_step(self, batch_data, batch_idx):
        token_ids = batch_data.token_ids
        label_ids = batch_data.label_ids
        embedding = self.embedding(token_ids)
        x = embedding.permute(0, 2, 1)
        out = [conv(x).unsqueeze(2) for conv in self.conv1ds]
        out = torch.cat(out, dim=2)
        out = out.view(x.size(0), -1)
        loss = self.loss_func(out, label_ids)
        return {"loss": loss, "logits": out}

    def validation_step(self, batch_data, batch_idx):
        token_ids = batch_data.token_ids
        label_ids = batch_data.label_ids
        embedding = self.embedding(token_ids)
        x = embedding.permute(0, 2, 1)
        out = [conv(x).unsqueeze(2) for conv in self.conv1ds]
        out = torch.cat(out, dim=2)
        out = out.view(x.size(0), -1)
        loss = self.loss_func(out, label_ids)
        pred = torch.argmax(out, dim=-1)
        val_f1 = f1_score(y_true=label_ids.detach().numpy(), y_pred=pred.detach().numpy(),average="micro")
        self.log("val_f1", val_f1)
        return {"loss": loss, "val_f1": val_f1}

