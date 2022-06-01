# -*- coding: utf-8 -*-
import datetime
import math
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from transformers import *

import utils_ucca
from vocab import Vocab
from dataset import UCCADataset
from ucca_parser import UCCA_Parser
from torch.utils.data import Dataset, DataLoader
from evaluator import UCCA_Evaluator
from trainer import Trainer

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

CUDA_LAUNCH_BLOCKING = 1

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

#parameters
config_path = 'config.json'
config = utils_ucca.get_config(config_path)

if config.model_name == "bert":
    model = BertModel.from_pretrained(
        config.model_path,
        output_hidden_states=True,
        output_attentions=True).to(device)
    tokenizer = BertTokenizer.from_pretrained(config.model_path)


# input_ids = torch.tensor(tokenizer.encode("This is an example",  add_special_tokens=False)).unsqueeze(0)  # Batch size 1
# print(input_ids.shape)
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
 
# print(last_hidden_states.shape)
#token_embeddings = torch.stack(encoded_layers, dim=0)
vocab = Vocab(utils_ucca.read_passages(config.train_path))

train_dataset = DataLoader(UCCADataset(config.train_path, tokenizer, vocab, train=True), batch_size=config.batch_size, shuffle=True, collate_fn=utils_ucca.collate_fn)
dev_dataset = DataLoader(UCCADataset(config.train_path, tokenizer, vocab, train=True), batch_size=config.batch_size, shuffle=True, collate_fn=utils_ucca.collate_fn)
test_dataset = DataLoader(UCCADataset(config.train_path, tokenizer, vocab, train=False), batch_size=config.batch_size, shuffle=False, collate_fn=utils_ucca.collate_fn)

ucca_parser = UCCA_Parser(device, model, vocab, config).to(device)

optimizer = optim.Adam(ucca_parser.parameters(), lr=config.lr)

ucca_evaluator = UCCA_Evaluator(
    device = device,
    parser=ucca_parser,
    gold_dic=config.dev_path,
)

trainer = Trainer(
    device=device,
    parser=ucca_parser,
    optimizer=optimizer,
    evaluator=ucca_evaluator,
    batch_size=config.batch_size,
    epoch=config.epoch,
    patience=config.patience,
    path=config.save_path,
)

trainer.train(train_dataset, dev_dataset)

ucca_evaluator.compute_accuracy(test_dataset, train=False, path= "dataset/results/")