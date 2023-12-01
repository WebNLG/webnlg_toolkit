import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.utils import resample
from torch.utils.data import DataLoader

from webnlg_toolkit.utils.data import load_webnlg_dataset


class RDF2TextDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, params=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters(params)

    def setup(self, stage):

        # handle multiple data files
        if "," in self.hparams.train_file:
            self.hparams.train_file = self.hparams.train_file.split(",")
        if "," in self.hparams.val_file:
            self.hparams.val_file = self.hparams.val_file.split(",")

        mt = self.has_param("multilingual")

        # read and prepare input data
        if isinstance(self.hparams.train_file, str):
            self.train = load_webnlg_dataset(self.hparams.train_file, lang=self.hparams.lang, 
                                             task=self.hparams.task, multilingual=mt, training=True)
            self.valid = load_webnlg_dataset(self.hparams.val_file, lang=self.hparams.lang, 
                                             task=self.hparams.task,  multilingual=mt, training=True)
        else:
            for f in self.hparams.train_file:
                if self.train is None:
                    self.train = load_webnlg_dataset(f, lang=self.hparams.lang, task=self.hparams.task, 
                                                     multilingual=mt, training=True)
                else:
                    self.train += load_webnlg_dataset(f, lang=self.hparams.lang, task=self.hparams.task, 
                                                      multilingual=mt, training=True)
            for f in self.hparams.val_file:
                if self.valid is None:
                    self.valid = load_webnlg_dataset(f, lang=self.hparams.lang, task=self.hparams.task, 
                                                     multilingual=mt, training=True)
                else:
                    self.valid += load_webnlg_dataset(f, lang=self.hparams.lang, task=self.hparams.task, 
                                                      multilingual=mt, training=True)

        print("All data loaded.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True, 
                            num_workers=self.hparams.train_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.hparams.batch_size, shuffle=False,
                            num_workers=self.hparams.val_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=1, 
                            pin_memory=False, collate_fn=self.prepro_collate)

    def prepro_collate(self, batch):
        inputs = [x[0] for x in batch]
        labels = [x[-1] for x in batch] if len(batch[0]) > 1 and isinstance(batch[0][-1], str) else []
        seqs = self.tokenizer(inputs+labels, max_length=self.hparams.max_length, padding=True, 
                              truncation=True, add_special_tokens=True, return_tensors='pt')

        input_ids = seqs.input_ids[:len(inputs)]
        input_mask = seqs.attention_mask[:len(inputs)]
        labels = seqs.input_ids[len(labels):] if labels != [] else None
        if labels is not None:
            labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation

        data = {}
        data["input_ids"] = input_ids
        data["attention_mask"] = input_mask
        data["labels"] = labels

        return data
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # only send tensors to device
        batch = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def get_param(self, param):
        if self.has_param(param):
            return self.hparams[param]
