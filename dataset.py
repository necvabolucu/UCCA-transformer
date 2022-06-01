import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import Parameter
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import utils_ucca

class UCCADataset(Dataset):
    def __init__(self, path, tokenizer, vocab, train=False):
        self.passages = utils_ucca.create_dataset(path, False)
        self.trees = [utils_ucca.generate_tree(passage).convert() for passage in self.passages]
        self.remotes = [utils_ucca.gerenate_remote(passage) for passage in self.passages]
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.train = train
        
    def __len__(self):
        return len(self.passages)
    
    def __getitem__(self, index):
        passage = self.passages[index]
        
        terminals = [
            (node.text, node.extra["pos"], node.extra["dep"], node.extra["ent_type"], node.extra["ent_iob"])
            for node in sorted(passage.layer("0").all, key=lambda x: x.position)
        ]
        
        words, pos, dep, ent_type, ent_iob = zip(*terminals)
        words = list(words)
        pos = list(pos)
        dep = list(dep)
        ent = list(ent_type)
        ent_iob = list(ent_iob)
        tree = self.trees[index]
        
        words = [self.vocab.START] + words + [self.vocab.STOP]
        _word_idxs = self.vocab.word2id([self.vocab.START] + words + [self.vocab.STOP])
        _pos_idxs = self.vocab.pos2id([self.vocab.START] + pos + [self.vocab.STOP])
        _dep_idxs = self.vocab.dep2id([self.vocab.START] + dep + [self.vocab.STOP])
        _entity_idxs = self.vocab.entity2id([self.vocab.START] + ent + [self.vocab.STOP])
        _iob_idxs = self.vocab.ent_iob2id([self.vocab.START] + ent_iob + [self.vocab.STOP])

        nodes, (heads, deps, labels) = self.remotes[index]
        if len(heads) == 0:
            _remotes = ()
        else:
            heads, deps = torch.tensor(heads), torch.tensor(deps)
            labels = [[self.vocab.edge_label2id(l) for l in label] for label in labels]
            labels = torch.tensor(labels)
            _remotes = (heads, deps, labels)

        

        if not self.train:
            tree = []
            nodes = []
            _remotes = []
            


        return (torch.tensor(self.tokenizer.encode(" ".join(words),  add_special_tokens=False)),
                torch.tensor(_word_idxs),
                torch.tensor(_pos_idxs),
                torch.tensor(_dep_idxs),
                torch.tensor(_entity_idxs),
                torch.tensor(_iob_idxs),
                passage,
                tree,
                nodes,
                _remotes)


        
    

   