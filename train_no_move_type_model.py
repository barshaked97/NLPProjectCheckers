from mingpt.utils import set_seed
set_seed(44)
import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
# from data import get_othello
# from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDatasetNoMoveType
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

train_dataset = CharDatasetNoMoveType(kings=True, test=False)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

max_epochs = 5
# initialize a trainer instance and kick off training
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs,
    batch_size=24*1, # num batches per gpu x num gpus
    learning_rate=5e-4,
    lr_decay=True,
    warmup_tokens=len(train_dataset)*train_dataset.block_size*5,
    progress=0, #number of epochs already done
    final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
    num_workers=0,
    # ckpt_path=f"./ckpts/gpt_at{t_start}.ckpt",
    ckpt_path=f"./ckpts/gpt_no_move_type.ckpt"
)
trainer = Trainer(model, train_dataset, None, tconf)
device = trainer.device
print(t_start)
trainer.train()