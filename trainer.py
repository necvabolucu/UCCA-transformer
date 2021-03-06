import datetime
import math
import os
import time

import torch
import torch.nn as nn

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

class Trainer(object):
    def __init__(self, device, parser, optimizer, evaluator, batch_size, epoch, patience, path):
        self.device = device
        self.parser = parser
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.save_path = os.path.join(path, "parser.pt")
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience


    def update(self, batch):
        self.optimizer.zero_grad()
        self.parser.zero_grad()
        span_losses, remote_losses = 0, 0
        words, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote = (
            batch
        )
        
        batch_size = len(word_idxs)
        words = torch.split(words, 5, dim=0)
        word_idxs = torch.split(word_idxs, 5, dim=0)
        pos_idxs = torch.split(pos_idxs, 5, dim=0)
        dep_idxs = torch.split(dep_idxs, 5, dim=0)
        ent_idxs = torch.split(ent_idxs, 5, dim=0)
        ent_iob_idxs = torch.split(ent_iob_idxs, 5, dim=0)
        for i, word, word_idx, pos_idx, dep_idx, ent_idx, ent_iob_idx, in zip(range(0, batch_size, 5), words, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs):

            span_loss, remote_loss = self.parser.parse(
                word.to(self.device),
                word_idx.to(self.device),
                pos_idx.to(self.device),
                dep_idx.to(self.device),
                ent_idx.to(self.device),
                ent_iob_idx.to(self.device),
                passages[i : i + 5],
                trees[i : i + 5],
                all_nodes[i : i + 5],
                all_remote[i : i + 5],
            )

            span_losses += sum(span_loss)
            remote_losses += sum(remote_loss)
        loss = span_losses / batch_size + remote_losses
        loss.backward()
        nn.utils.clip_grad_norm_(self.parser.parameters(), 5.0)
        self.optimizer.step()
        return loss


    def train(self, train, dev):
        best_epoch, best_f, patience = 0, 0, 0

        print("start to train the model ")
        for e in range(1, self.epoch + 1):
            epoch_start_time = time.time()

            self.parser.train()
            time_start = datetime.datetime.now()

            for step, batch in enumerate(train):
                loss = self.update(batch)
                print(
                    "epoch %d batch %d/%d batch-loss %f epoch-elapsed %s "
                    % (
                        e,
                        step + 1,
                        int(math.ceil(len(train.dataset) / self.batch_size)),
                        loss,
                        format_elapsed(epoch_start_time),
                    )
                )

            f = self.evaluator.compute_accuracy(dev, train=True)
            if hasattr(self.optimizer, "schedule"):
                self.optimizer.schedule(f)
                
            if f >= best_f:
                patience = 0
                best_epoch = e
                best_f = f
                print("save the model...")
                print("the current best f is %f" % (best_f))
                self.parser.save(self.save_path)
            else:
                patience += 1

            time_end = datetime.datetime.now()
            print("epoch executing time is " + str(time_end - time_start) + "\n")
            if patience > self.patience:
                break

        print("train finished with epoch: %d / %d" % (e, self.epoch))
        print("the best epoch is %d , the best F1 on dev is %f" % (best_epoch, best_f))
        print(str(datetime.datetime.now()))
        self.evaluator.remove_temp()