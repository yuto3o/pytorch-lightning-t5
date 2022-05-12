# Pytorch Lightning T5

The Practice of Fine-tuning T5 Based on ⚡ PyTorch-lightning and 🤗 Hugging-face-Transformers

## Quick Start

### Preliminary

```shell
pip install -r requirements.txt
```

### Fine-tune T5 easily

```python

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from lightningt5 import *
from pytorch_lightning import Trainer

# init model
config = T5Config.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small', config=config)
tokenizer = T5Tokenizer.from_pretrained('t5-small', config=config)

# load data
toy_data = {'train': [{'source': ..., 'target': ...}, ...],
            'valid': [{'source': ..., 'target': ...}, ...],
            'test': [{'source': ..., 'target': ...}, ...]}

# callback
trainer = Trainer()

# init module
module = LightningT5Module(model, tokenizer, **vars(LightningT5ModuleArgument()))
dataset = LightningT5DataModuel(toy_data, tokenizer, **vars(LightningT5DataModuleArgument()))

# training
trainer.fit(module, dataset)
```

### Fine-tune T5 with CLI and Powerful Trainer

```shell
python cil.py --device auto --accelerator auto --val_check_interval 0.25 --num_workers 8 --max_epoch 5 ...
```