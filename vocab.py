import utils_ucca
import io
import pickle
from collections import Counter
from itertools import chain
from convert import InternalParseNode

class Vocab(object):
    def __init__(self, passages):
        self.passages = passages
        self.UNK = "<UNK>"
        self.START = "<START>"
        self.STOP = "<STOP>"
        self.PAD = "<PAD>"
        self.NULL = "<NULL>" 
        
        word, pos, dep, ent, ent_iob, edge_label, parse_label = self.collect()
        
        self._word = [self.PAD, self.START, self.STOP, self.UNK] + word
        self._pos = [self.PAD] + pos + [self.START, self.STOP]
        self._dep = [self.PAD] + dep + [self.START, self.STOP]
        self._ent = [self.PAD] + ent + [self.START, self.STOP]
        self._ent_iob = [self.PAD] + ent_iob + [self.START, self.STOP]
        
        self.num_train_word = len(self._word)

        self._edge_label = [self.NULL] + edge_label
        self._parse_label = [()] + parse_label

        self._word2id = {w: i for i, w in enumerate(self._word)}
        self._pos2id = {p: i for i, p in enumerate(self._pos)}
        self._dep2id = {p: i for i, p in enumerate(self._dep)}
        self._ent2id = {p: i for i, p in enumerate(self._ent)}
        self._ent_iob2id = {p: i for i, p in enumerate(self._ent_iob)}
        
        self._edge_label2id = {e: i for i, e in enumerate(self._edge_label)}
        self._parse_label2id = {p: i for i, p in enumerate(self._parse_label)}    
    
    def collect(self):
        token, edge = [], []
        pos, dep, ent, ent_iob = [], [], [], []
        for passage in self.passages:
            for node in passage.layer("0").all:
                token.append(node.text)
                pos.append(node.extra["pos"])
                dep.append(node.extra["dep"])
                ent.append(node.extra["ent_type"])
                ent_iob.append(node.extra["ent_iob"])
            for node in passage.layer("1").all:
                for e in node._incoming:
                    if e.attrib.get("remote"):
                        edge.append(e.tag)

        words, edge_label = sorted(set(token)), sorted(set(edge))
        pos, dep, ent, ent_iob = sorted(set(pos)), sorted(set(dep)), sorted(set(ent)), sorted(set(ent_iob))
        
        parse_label = []
        for passage in self.passages:
            tree = utils_ucca.generate_tree(passage).convert()
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, InternalParseNode):
                    parse_label.append(node.label)
                    nodes.extend(reversed(node.children))
        parse_label = sorted(set(parse_label))

        return words, pos, dep, ent, ent_iob, edge_label, parse_label

    @property
    def PAD_index(self):
        return self._word2id[self.PAD]

    @property
    def STOP_index(self):
        return self._word2id[self.STOP]

    @property
    def UNK_index(self):
        return self._word2id[self.UNK]

    @property
    def NULL_index(self):
        return self._parse_label2id[()]

    @property
    def num_word(self):
        return len(self._word)

    @property
    def num_pos(self):
        return len(self._pos)

    @property
    def num_dep(self):
        return len(self._dep)

    @property
    def num_ent(self):
        return len(self._ent)

    @property
    def num_ent_iob(self):
        return len(self._ent_iob)

    @property
    def num_edge_label(self):
        return len(self._edge_label)

    @property
    def num_parse_label(self):
        return len(self._parse_label)


    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def __repr__(self):
        return "word:%d, pos:%d, dep:%d, ent:%d, ent_iob:%d, edge_label:%d, parse_label:%d" % (
            self.num_word,
            self.num_pos,
            self.num_dep,
            self.num_ent,
            self.num_ent_iob,
            self.num_edge_label,
            self.num_parse_label,
        )

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            word_idx = self._word2id.get(word, self.UNK_index)
            return word_idx
        elif isinstance(word, list):
            word_idxs = [self._word2id.get(w, self.UNK_index) for w in word]
            return word_idxs

    def pos2id(self, pos):
        assert isinstance(pos, str) or isinstance(pos, list)
        if isinstance(pos, str):
            return self._pos2id.get(pos, 0)  # if pos not in training data, index to 0 ?
        elif isinstance(pos, list):
            return [self._pos2id.get(l, 0) for l in pos]

    def dep2id(self, dep):
        assert isinstance(dep, str) or isinstance(dep, list)
        if isinstance(dep, str):
            return self._dep2id.get(dep, 0)
        elif isinstance(dep, list):
            return [self._dep2id.get(l, 0) for l in dep]

    def entity2id(self, entity):
        assert isinstance(entity, str) or isinstance(entity, list)
        if isinstance(entity, str):
            return self._ent2id.get(entity, 0)
        elif isinstance(entity, list):
            return [self._ent2id.get(l, 0) for l in entity]

    def ent_iob2id(self, iob):
        assert isinstance(iob, str) or isinstance(iob, list)
        if isinstance(iob, str):
            return self._ent_iob2id.get(iob, 0)
        elif isinstance(iob, list):
            return [self._ent_iob2id.get(l, 0) for l in iob]

    def edge_label2id(self, label):
        if isinstance(label, str):
            return self._edge_label2id.get(label, 0)
        else:
            return [self._edge_label2id.get(l, 0) for l in label]
    
    def id2parse_label(self, id):
        return self._parse_label[id]

    def id2edge_label(self, id):
        return self._edge_label[id]

    def parse_label2id(self, label):
        return self._parse_label2id[label]        
        
        