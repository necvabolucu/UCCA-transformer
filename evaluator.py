import os
import subprocess
import torch
from ucca.convert import passage2file
import tempfile
from argparse import ArgumentParser
from itertools import repeat

from ucca import evaluation, constructions, ioutil



    
def write_passages(dev_predicted, path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in dev_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


class UCCA_Evaluator(object):
    def __init__(
        self, device, parser, gold_dic=None, pred_dic=None,
    ):
        self.device = device
        self.parser = parser
        self.gold_dic = gold_dic
        self.pred_dic = pred_dic
        self.temp_pred_dic = tempfile.TemporaryDirectory(prefix="ucca-eval-")
        self.best_F = 0

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        predicted = []
        for batch in loader:
            words, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote = batch
            words = words.to(self.device)
            word_idxs = word_idxs.to(self.device)
            pos_idxs = pos_idxs.to(self.device)
            dep_idxs = dep_idxs.to(self.device)
            ent_idxs = ent_idxs.to(self.device)
            ent_iob_idxs = ent_iob_idxs.to(self.device)

            pred_passages = self.parser.parse(words, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages)
            predicted.extend(pred_passages)
        return predicted
        
    def remove_temp(self):
        self.temp_pred_dic.cleanup()

    def compute_accuracy(self, loader, train=False, path=None):
        passage_predicted = self.predict(loader)
        if not train:
            write_passages(passage_predicted, path)

        child = subprocess.Popen(
            "python -m scripts.evaluate_standard {} {} -f".format(
                self.gold_dic, passage_predicted
            ),
            shell=True,
            stdout=subprocess.PIPE,
        )
        eval_info = str(child.communicate()[0], encoding="utf-8")
        try:
            Fscore = eval_info.strip().split("\n")[-1]
            Fscore = Fscore.strip().split()[-1]
            Fscore = float(Fscore)
            print("Fscore={}".format(Fscore))
        except IndexError:
            print("Unable to get FScore. Skipping.")
            Fscore = 0

        if Fscore > self.best_F:
            print('\n'.join(eval_info.split('\n')[1:]))
            self.best_F = Fscore
            if self.pred_dic:
                write_passages(passage_predicted, self.pred_dic)
        return Fscore