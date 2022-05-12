import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import HfArgumentParser
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

from lightningt5 import *


def main(module_args, data_module_args, trainer_args):
    # init model
    config = T5Config.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small', config=config)
    tokenizer = T5Tokenizer.from_pretrained('t5-small', config=config)

    # load data
    toy_data = [json.loads(record) for record in open('data/bill_sum_ca_test.jsonl')]
    toy_data = [{'source': record['text'], 'target': record['summary']} for record in toy_data]

    # callback
    callbacks = [ModelCheckpoint(monitor='valid_loss',
                                 filename='{epoch:02d}-{step:09d}-{valid_loss:.4f}',
                                 save_top_k=10,
                                 mode='max'),
                 LearningRateMonitor('step')]
    trainer = Trainer.from_argparse_args(trainer_args,
                                         callbacks=callbacks)

    # init module
    module = LightningT5Module(model, tokenizer, **vars(module_args))
    dataset = LightningT5DataModuel({'train': toy_data[:-100], 'valid': toy_data[-100:-50], 'test': toy_data[-50:]},
                                    tokenizer, **vars(data_module_args))

    # training
    seed_everything(42, workers=True)
    trainer.fit(module, dataset)


if __name__ == '__main__':
    parser = HfArgumentParser([LightningT5ModuleArgument, LightningT5DataModuleArgument])
    parser = Trainer.add_argparse_args(parser)

    parser.parse_args_into_dataclasses()
    module_args, data_module_args, trainer_args = parser.parse_args_into_dataclasses()
    main(module_args, data_module_args, trainer_args)
