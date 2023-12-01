import argparse

import torch
import pandas as pd
from tqdm import tqdm
from pytorch_lightning import LightningModule
from transformers import (
    #AdamW,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW # using this instead of transformers one because of the bug in the latter

from torch.utils.data import DataLoader

from webnlg_toolkit.utils.data import load_webnlg_dataset
from webnlg_toolkit.data import RDF2TextDataModule
from webnlg_toolkit.eval.eval import run as run_eval
from webnlg_toolkit.eval.eval import print_results

SPECIAL_TOKENS = ["<S>", "<P>", "<O>"]
INF_PARAMS = ["multilingual", "lang", "base_model", "max_length", "task"]
RP = {
    "en": 1.0,
    "ru": 1.0,
    "br": 3.5,
    "cy": 3.5,
    "ga": 3.5,
    "mt": 3.5,
}


def load_model(model_ckpt, tokenizer=None, device="cuda", **kwargs):
    if model_ckpt.endswith(".ckpt"):
        model = T5Module.load_from_checkpoint(model_ckpt, **kwargs).to(device).eval()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, return_dict=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    # unpack model and accompanying components
    hparams = {}
    if isinstance(model, T5Module):
        # extract what we need from PytorchLightning checkpoint before reducing to HF model
        tokenizer = model.tokenizer
        hparams = model.hparams
        model.model.config.update(hparams) # directly update internal model config
        model = model.model
    else:
        hparams = {p: getattr(model.config, p, None) for p in INF_PARAMS}

    return model, tokenizer, hparams


def inference(model_ckpt, test_file, lang="en", out_file=None, training=False, batch_size=16, 
              max_batches=None, do_eval=False, metrics="bleu,chrf++,ter,bert", device="cuda", **kwargs):

    model, tokenizer, hparams = load_model(model_ckpt, device=device, **kwargs)
    dm = RDF2TextDataModule(tokenizer, params=hparams)

    mlm = hparams.get("multilingual", False)
    task = hparams.get("task", "rdf2text")

    data = load_webnlg_dataset(test_file, lang=lang, training=training, multilingual=mlm, task=task)
    test_data = DataLoader(data, batch_size=batch_size, collate_fn=dm.prepro_collate)

    out_texts = []
    for i, batch in enumerate(tqdm(test_data)):
        # ignore labels for inference
        batch = {k: xi.to(device, non_blocking=True) for k, xi in batch.items() if k != "labels"}
        generated_ids = model.generate(
            **batch,
            use_cache=True,
            decoder_start_token_id=None, # None is default and so will be handled internally
            num_beams=4,
            max_length=128,
            repetition_penalty=RP[lang],
        )
        skip_special = (task == "rdf2text") # skip special tokens for semantic parsing
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special, 
                                          clean_up_tokenization_spaces=True)
        
        if not skip_special:
            # strip non-task special tokens
            gen_text = [x.replace("</s>", "").replace("<pad>", "").replace("<unk>", "") for x in gen_text]

        out_texts += gen_text

        if max_batches is not None and not do_eval and i >= max_batches:
            break

    for i in range(len(out_texts)):
        data[i] += (out_texts[i],)
    data_df = pd.DataFrame(data, columns=["input", "ref", "output"])

    if do_eval:
        num_refs = max([len(row.ref) for _, row in data_df.iterrows()])
        label_seqs = [[y[0] for y in ys] for ys in data_df["ref"]]
        result = run_eval(refs_path=label_seqs, hyps_path=data_df["output"].tolist(), 
                          lng=lang, num_refs=num_refs, metrics=metrics)
        print_results(result, metrics=metrics, lng=lang)

        # save sentence-level scores to df
        for k, v in result.items():
            if isinstance(v, list):
                data_df[k] = v

    if out_file is not None:
        data_df.to_csv(out_file, index=False)
        print(f"Saved to {out_file}.")

    return data_df


class T5Module(LightningModule):
    def __init__(self, model_name_or_path="google/mt5-base", tokenizer=None, params=None, **kwargs):
        super().__init__()

        self.save_hyperparameters(params)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
            self.add_new_tokens()
        else:
            self.tokenizer = tokenizer

        # training loss cache to log mean every n steps
        self.train_losses = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch, use_cache=False)
        loss = outputs[0]

        # logging mean loss every `n` steps
        if batch_idx % max(1, int(self.hparams.train_check_interval * self.trainer.num_training_batches)) == 0 and len(self.train_losses) > 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_loss", avg_loss, sync_dist=True)
            self.train_losses = []

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch, use_cache=False)
        val_loss, logits = outputs[:2]

        return {"loss": val_loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def add_new_tokens(self):
        self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def save_tokenizer(self, path):
        self.tokenizer.save_pretrained(path)
        print(f"Tokenizer saved to {path}.")

    def save_model(self, path):
        # add inference parameters to model config
        self.model.config.update({p: self.hparams.get(p) for p in INF_PARAMS})

        self.model.save_pretrained(path)
        self.save_tokenizer(path)
        print(f"{type(self.model)} model saved to {path}.")

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--train_file", type=str, default=None, required=False)
        parser.add_argument("--val_file", type=str, default=None, required=False)
        parser.add_argument("--train_check_interval", type=float, default=0.01)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--lang", type=str, default="en", required=False)
        parser.add_argument("--base_model", type=str, default="google/mt5-base", required=False)
        parser.add_argument("--task", type=str, default="rdf2text", required=False)
        parser.add_argument("--multilingual", action="store_true")

        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--ckpt_metric", type=str, default="val_loss", required=False,)
        parser.add_argument("--hidden_dropout_prob", type=float, default=None, required=False,)

        return parser
