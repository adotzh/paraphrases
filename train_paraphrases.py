import argparse
from typing import Dict
import os
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers import AutoModel, AutoTokenizer, AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

os.environ["TOKENIZERS_PARALLELISM"] = "0"

class Embedder(nn.Module):
    def __init__(self, model_path, freeze_bert, layer_num):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_path)
        self.model.trainable = not freeze_bert
        self.bert_dim = self.model.config.hidden_size
        self.layer_num = layer_num

    def forward(self, input_ids, attention_mask):
        output = self.model(
           input_ids,
           attention_mask=attention_mask,
           return_dict=True,
           output_hidden_states=True
        )
        layer_embeddings = output.hidden_states[self.layer_num]
        embeddings = torch.mean(layer_embeddings, dim=1)
        norm = embeddings.norm(p=2, dim=1, keepdim=True)
        embeddings = embeddings.div(norm)
        return embeddings


class ClusteringContrastiveModel(LightningModule):
    def __init__(self, model_path, freeze_bert=False, layer_num=-1, margin=0.5, lr=1e-5):
        super().__init__()

        self.embedder = Embedder(
            model_path,
            freeze_bert=freeze_bert,
            layer_num=layer_num
        )
        self.lr = lr
        self.loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, left, right, labels):
        left_embeddings = self.embedder(left["input_ids"], left["attention_mask"])
        right_embeddings = self.embedder(right["input_ids"], right["attention_mask"])
        loss = self.loss(left_embeddings, right_embeddings, labels)
        return loss

    def training_step(self, batch, batch_nb):
        train_loss = self(*batch)
        return train_loss

    def validation_step(self, batch, batch_nb):
        val_loss = self(*batch)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return [optimizer]


class NewsDataset(Dataset):
    def __init__(self, records, model_path, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        self.max_tokens = max_tokens
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        fields = self.records[index]
        label = fields[-1]
        samples = fields[:-1]
        samples = [self.tokenizer(
            s,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ) for s in samples]
        samples = [{key: value.squeeze(0) for key, value in s.items()} for s in samples]
        return samples[0], samples[1], torch.tensor(float(1 if label == 1 else -1))


def form_pairs(filename):
    records = []
    with open(filename, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append((record["s1"], record["s2"], record["target"]))
    return records


def train_paraphrases(
    initial_model_name,
    train_path,
    max_tokens,
    out_dir,
    batch_size,
    grad_accum_steps,
    epochs,
    lr
):
    records = form_pairs(train_path)
    random.shuffle(records)
    border = int(len(records) * 0.8)
    train_records = records[:border]
    val_records = records[border:]

    train_data = NewsDataset(train_records, initial_model_name, max_tokens)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = NewsDataset(val_records, initial_model_name, max_tokens)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    LOG_EVERY_N_STEPS = 5
    model = ClusteringContrastiveModel(initial_model_name, lr=lr)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min"
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=True,
        accumulate_grad_batches=grad_accum_steps,
        max_epochs=epochs,
        callbacks=[early_stop_callback],
        log_every_n_steps=LOG_EVERY_N_STEPS
    )
    trainer.fit(model, train_loader, val_loader)
    train_data.tokenizer.save_pretrained(out_dir)
    model.embedder.model.save_pretrained(out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-model-name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-05)
    args = parser.parse_args()
    train_paraphrases(**vars(args))
