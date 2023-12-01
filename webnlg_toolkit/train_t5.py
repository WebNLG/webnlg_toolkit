import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from webnlg_toolkit.data import RDF2TextDataModule
from webnlg_toolkit.t5 import T5Module


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()

    # add model specific args to parser
    parser = T5Module.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and finetuner class
    if args.checkpoint is None:
        model = T5Module(args.base_model, params=args)
    else:
        model = T5Module.load_from_checkpoint(args.checkpoint, params=args, strict=False)

    dm = RDF2TextDataModule(model.tokenizer, params=args)

    logger = CSVLogger("logs", name=args.name)
    checkpoint_callback = None
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir)

    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=logger,
        accelerator="gpu",
        # strategy="ddp",
        strategy="ddp_find_unused_parameters_false",
        #plugins=DDPStrategy(find_unused_parameters=True),
        callbacks=checkpoint_callback,
        precision=32,)

    trainer.fit(model, dm)
