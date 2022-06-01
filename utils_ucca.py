import os
import copy
from ucca.convert import to_text, xml2passage
from ucca import textutil
from convert import UCCA2tree
from ucca.layer1 import FoundationalNode
from ucca.layer0 import Terminal
import json
from torch.nn.utils.rnn import pad_sequence
from argparse import Namespace

def get_config(config_filepath):
    with open(config_filepath, "r") as config_file:
        conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
    return conf
   
    
def read_passages(path):
    passages = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            print(file_path)
        pas = xml2passage(file_path)
        passages.append(pas)
    return passages    

def create_dataset(path, feature=False):
    # If the dataset does not contain features (pos tag, dependency etc.)
    passages = read_passages(path)
    if feature:
        passages = list(textutil.annotate_all(passages))
        
    return passages

def generate_tree(passage):
    temp_passage = copy.deepcopy(passage)
    if "1" in passage._layers:
        return UCCA2tree(temp_passage)
    else:
        return None
    
def gerenate_remote(passage):
    def get_span(node):
        children = [i.child for i in node.outgoing if not i.attrib.get("remote")]
        terminals = [t for c in children for t in c.get_terminals()]
        terminals = list(sorted(terminals, key=lambda x: x.position))
        # terminals = node.get_terminals()
        return (terminals[0].position - 1, terminals[-1].position)

    if "1" not in passage._layers:
        return [], ([], [], [])
    edges, spans = [], []
    nodes = [
        node
        for node in passage.layer("1").all
        if isinstance(node, FoundationalNode) and not node.attrib.get("implicit")
    ]
    ndict = {node: i for i, node in enumerate(nodes)}
    spans = [get_span(i) for i in nodes]

    remote_nodes = []
    for node in nodes:
        for i in node.incoming:
            if i.attrib.get("remote"):
                remote_nodes.append(node)
                break
    heads = [[ndict[n]] * len(nodes) for n in remote_nodes]
    deps = [list(range(len(nodes))) for _ in remote_nodes]
    labels = [["<NULL>"] * len(nodes) for _ in remote_nodes]
    for id, node in enumerate(remote_nodes):
        for i in node.incoming:
            if i.attrib.get("remote"):
                labels[id][ndict[i.parent]] = i.tag

    return spans, (heads, deps, labels)

def collate_fn(data):
    words, word_idx, pos_idx, dep_idx, ent_idx, ent_iob_idx, passages, trees, all_nodes, all_remote = zip(*data)
    return (
            pad_sequence(words, True),
            pad_sequence(word_idx, True),
            pad_sequence(pos_idx, True),
            pad_sequence(dep_idx, True),
            pad_sequence(ent_idx, True),
            pad_sequence(ent_iob_idx, True),
            passages,
            trees,
            all_nodes,
            all_remote,
   )