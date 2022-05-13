# -*- coding: utf-8 -*-
import inspect
import logging
import os
import pickle
from collections import abc
from dataclasses import dataclass, field
from pprint import pformat
from typing import List, Dict

import torch
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import Dataset as _Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass()
class LightningT5ModuleArgument:
    learning_rate: float = field(default=0.001, metadata={'help': 'learning rate'})
    num_warmup_steps: int = field(default=0, metadata={'help': 'warmup steps'})


class LightningT5Module(LightningModule):

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 **training_args):
        super(LightningT5Module, self).__init__()
        self.save_hyperparameters(training_args)

        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, labels: torch.LongTensor,
                **kwargs):
        # prepare decoder_input_ids
        # If has the *prepare_decoder_input_ids_from_labels*, use it to prepare the *decoder_input_ids*
        if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            decoder_input_ids = decoder_input_ids
        else:
            decoder_input_ids = None

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          labels=labels)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_size):
        loss = self(**batch).loss
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_size):
        loss = self(**batch).loss
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        n = len(self.trainer.datamodule.train_dataloader())
        total = self.trainer.max_epochs * n // (
                max(1,
                    self.trainer.num_devices if self.trainer.gpus else 0) * self.trainer.accumulate_grad_batches)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.num_warmup_steps,
                                                    num_training_steps=total)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def predict(self, text: List[str],
                min_length: int = 0,
                max_length: int = 512,
                num_return_sequences: int = 1,
                num_beams: int = 5,
                top_k: int = None,
                top_p: float = None,
                repetition_penalty: float = None,
                length_penalty: float = None,
                early_stopping: bool = True,
                skip_special_tokens: bool = True,
                clean_up_tokenization_spaces: bool = True):
        r"""

        Args:
            text: Sequence[str], input text, (batch_size,)
            min_length: int, min output text length
            max_length: int, max output text length

            read more details from transformers.generation_utils.GenerationMixin.generate
            num_return_sequences: int
            num_beams: int
            top_k: int
            top_p: float
            repetition_penalty: float
            length_penalty: float
            early_stopping: bool

            read more details from tokenizer.batch_decode
            skip_special_tokens: bool
            clean_up_tokenization_spaces: bool

        Returns:
            [batch_size, num_return_sequences]
        """

        inputs = self.tokenizer(text,
                                padding=True,
                                return_tensors='pt')
        inputs = inputs.to(self.device)
        outputs = self.model.generate(input_ids=inputs.input_ids,
                                      attention_mask=inputs.attention_mask,
                                      num_beams=num_beams,
                                      min_length=min_length,
                                      max_length=max_length,
                                      repetition_penalty=repetition_penalty,
                                      length_penalty=length_penalty,
                                      early_stopping=early_stopping,
                                      top_p=top_p,
                                      top_k=top_k,
                                      num_return_sequences=num_return_sequences,
                                      output_scores=True,
                                      return_dict_in_generate=True
                                      )

        ans = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces)

        in_b, _ = inputs.input_ids.shape
        group_ans = [ans[i * num_return_sequences:(i + 1) * num_return_sequences] for i in range(in_b)]

        return group_ans


@dataclass()
class LightningT5DataModuleArgument:
    source_max_length: int = field(default=512, metadata={'help': 'source max length'})
    target_max_length: int = field(default=512, metadata={'help': 'target max length'})

    train_batch_size: int = field(default=16, metadata={'help': 'training batch size'})
    valid_batch_size: int = field(default=None, metadata={'help': 'validating batch size'})
    test_batch_size: int = field(default=None, metadata={'help': 'testing batch size'})

    num_workers: int = field(default=0, metadata={'help': 'number of workers'})


class LightningT5DataModuel(LightningDataModule):

    def __init__(self,
                 splits: Dict[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 **datamodule_args):
        """

        Args:
            splits: Dict[str, List[Dict]]
                {'train': [
                        {'source': ..., 'target': ...},
                        ... ]
                 'valid': ...
                 'test': ...
                }
        """
        super(LightningT5DataModuel, self).__init__()
        self.save_hyperparameters(datamodule_args)

        self.splits_raw = splits
        self.tokenizer = tokenizer

        self.splits_encode = None

    def encode(self, source, target):
        # seq2seq task encoding
        inputs = self.tokenizer(source,
                                padding='max_length',
                                truncation=True,
                                return_token_type_ids=False,  # exclude token type ids
                                max_length=self.hparams.source_max_length)
        if target:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target,
                                        padding='max_length',
                                        truncation=True,
                                        return_token_type_ids=False,  # exclude token type ids
                                        max_length=self.hparams.target_max_length)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            labels['input_ids'] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in labels['input_ids']
            ]
            inputs['labels'] = labels['input_ids']

        inputs = inputs.convert_to_tensors('pt')
        return inputs

    def prepare_data(self) -> None:

        @cache(f'prepare_data.pkl')
        def tokenize(split):
            def _tokenize(data):
                out = []
                for instance in tqdm(data, desc='tokenize'):
                    out.append(self.encode(instance['source'], instance['target']))
                return out

            return {k: _tokenize(split[k]) for k in split}

        # summary
        logger.info(pformat({k: len(self.splits_raw[k]) for k in self.splits_raw}))
        self.splits_encode = tokenize(self.splits_raw)

    def collate_fn(self, batch):

        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, str):
            return batch
        elif isinstance(elem, abc.Sequence):
            it = iter(batch)
            elem_size = len(next(it))

            if not all(len(elem) == elem_size for elem in it):
                return batch

            return torch.LongTensor(batch)
        elif isinstance(elem, abc.Mapping):
            return elem_type({key: self.collate_fn([d[key] for d in batch]) for key in elem})
        elif isinstance(elem, int):
            return torch.LongTensor(batch)
        else:
            return NotImplementedError(elem_type)

    def train_dataloader(self):

        return DataLoader(Dataset(self.splits_encode['train']), batch_size=self.hparams.train_batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=True, drop_last=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):

        if 'valid' not in self.splits_encode:
            return None

        return DataLoader(Dataset(self.splits_encode['valid']),
                          batch_size=self.hparams.valid_batch_size if self.hparams.valid_batch_size else self.hparams.train_batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):

        if 'test' not in self.splits_encode:
            return None

        return DataLoader(Dataset(self.splits_encode['test']),
                          batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size else self.hparams.train_batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.hparams.num_workers)


class Dataset(_Dataset):

    def __init__(self, data):
        self.data_source = data

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, i):
        return self.data_source[i]


# code from FastNLP
def cache(_cache_fp, _refresh=False, _verbose=1):
    """cache wrapper"""

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError(
                    "The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):

            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


__all__ = ['LightningT5ModuleArgument', 'LightningT5Module', 'LightningT5DataModuel', 'LightningT5DataModuleArgument']
