import csv
import os
from argparse import ArgumentParser

from ucca import layer0, layer1, textutil
from ucca.ioutil import get_passages_with_progress_bar, write_passage
from ucca.normalization import fparent, remove, copy_edge, destroy
from ucca.textutil import annotate_all, Attr

desc = """Convert the English Wiki corpus from version 1.0 to 1.2"""


def get_annotation(terminal, attr):
    return terminal.extra[attr.key]


def is_main_relation(node):
    while node is not None and node.ftag == layer1.EdgeTags.Center:
        node = fparent(node)
    return node is not None and node.ftag in {layer1.EdgeTags.Process, layer1.EdgeTags.State}


def move_node(node, new_parent, tag=None):
    for edge in node.incoming:
        if edge.parent == fparent(edge):
            copy_edge(edge, parent=new_parent, tag=tag)
            remove(edge.parent, edge)
            break
    # for (parent_id, child_id), count in Counter((edge.parent.ID, edge.child.ID) for edge in new_parent).items():
    #     if count > 1:
    #         raise ValueError("There are %d edges from %s to %s" % (count, parent_id, child_id))


AUX = {"have", "be", "will", "to", "do", "'s", "'ve", "'ll", "'re", "'d", "'m"}


def extract_aux(terminal, parent, grandparent):
    if get_annotation(terminal, Attr.LEMMA) in AUX and is_main_relation(grandparent) and (
            parent.ftag == layer1.EdgeTags.Function or
            parent.ftag in {layer1.EdgeTags.Elaborator, layer1.EdgeTags.Relator} and
            get_annotation(terminal, Attr.DEP) in {"aux", "auxpass"}):
        move_node(parent, fparent(grandparent), tag=layer1.EdgeTags.Function)
        return True
    return False


LIGHT_VERBS = {"take", "make", "give", "have", "pay"}


def set_light_verb_function(terminal, parent, grandparent):
    if get_annotation(terminal, Attr.LEMMA) in LIGHT_VERBS and \
            is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Elaborator:
        if len(grandparent.centers) != 1 or len(grandparent.centers[0].get_terminals()) != 1 or \
                get_annotation(grandparent.centers[0].get_terminals()[0], Attr.DEP) != "ccomp":
            move_node(parent, grandparent, tag=layer1.EdgeTags.Function)
            return True
    return False


MODALS = {"can", "could", "may", "might", "shall", "should", "would", "must"}


def extract_modal(terminal, parent, grandparent):
    if (get_annotation(terminal, Attr.LEMMA) in MODALS or
        get_annotation(terminal, Attr.POS) in {"VERB", "ADV"} and
        get_annotation(terminal, Attr.DEP) not in {"aux", "auxpass"}) and \
            is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Elaborator:
        move_node(parent, fparent(grandparent), tag=layer1.EdgeTags.Adverbial)
        return True
    return False


def extract_relator(terminal, parent, grandparent):
    following_uncle = None
    for node in grandparent.iter():
        if node.tag == layer1.NodeTags.Foundational and node.start_position == 1 + terminal.position and \
                node.ftag in {layer1.EdgeTags.Participant, layer1.EdgeTags.Adverbial}:
            following_uncle = node
    if following_uncle is not None and is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Relator and \
            grandparent.end_position == terminal.position:
        move_node(parent, following_uncle)
        return True
    return False


def extract_that(terminal, parent, grandparent):
    del grandparent
    if get_annotation(terminal, Attr.LEMMA) == "that":
        following_scene = None
        for node in terminal.root.layer(layer1.LAYER_ID).heads[0].iter():
            if node.tag == layer1.NodeTags.Foundational and node.start_position == 1 + terminal.position and \
                    node.ftag == layer1.EdgeTags.ParallelScene:
                following_scene = node
        if following_scene is not None:
            move_node(parent, following_scene, tag=layer1.EdgeTags.Relator)
            return True
    return False


GROUND = {"seem", "feel", "sound", "taste", "look", "smell"}


def extract_ground(terminal, parent, grandparent):
    if get_annotation(terminal, Attr.LEMMA) in GROUND:
        if is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Elaborator:
            move_node(parent, fparent(grandparent), tag=layer1.EdgeTags.Ground)
            return True
    return False


def fix_punct(terminal, parent, grandparent):
    del grandparent
    if parent.tag == layer1.NodeTags.Foundational and terminal.tag == layer0.NodeTags.Punct and not \
            (len(parent.terminals) + len(parent.punctuation) == len(parent.children) > 1):  # Unanalyzable
        for edge in parent:
            parent.remove(edge)
        parent.root.layer(layer1.LAYER_ID).add_punct(parent, terminal)
        return True
    return False


def fix_root_terminal_child(terminal, parent, grandparent):
    del grandparent
    if not parent.incoming:
        f1 = parent.root.layer(layer1.LAYER_ID).add_fnode(parent, layer1.EdgeTags.Function)
        move_node(terminal, f1)
        return True
    return False


def fix_unary_participant(terminal, parent, grandparent):
    del terminal, parent
    while grandparent.incoming:
        if len(grandparent.outgoing) == 1 and grandparent.outgoing[0].tag == layer1.EdgeTags.Participant:
            for edge in grandparent.incoming:
                copy_edge(edge, child=grandparent.children[0])
            destroy(grandparent)
            return True
        grandparent = grandparent.parents[0]
    return False


def flag_relator_starts_main_relation(terminal, parent, grandparent):
    return grandparent.start_position == terminal.position and \
        is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Relator


def flag_suspected_secondary(terminal, parent, grandparent):
    return get_annotation(terminal, Attr.DEP) not in {"det"} and \
        is_main_relation(grandparent) and parent.ftag == layer1.EdgeTags.Elaborator


RULES = (extract_ground, extract_aux, set_light_verb_function, extract_modal, extract_relator, extract_that,
         fix_punct, fix_root_terminal_child, fix_unary_participant,
         flag_relator_starts_main_relation, flag_suspected_secondary)


def convert_passage(passage, report_writer):
    for rule in RULES:
        for terminal in passage.layer(layer0.LAYER_ID).all:
            parent = fparent(terminal)
            grandparent = fparent(parent)
            grandparent_str = str(grandparent)
            if len(parent.children) == 1 and rule(terminal, parent, grandparent):
                report_writer.writerow((rule.__name__, passage.ID, terminal.ID, get_annotation(terminal, Attr.POS),
                                        grandparent_str, fparent(fparent(terminal))))


def main(args):
    textutil.BATCH_SIZE = 1
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("rule", "passage", "terminal", "pos", "before", "after"))
        for passage in annotate_all(get_passages_with_progress_bar(args.passages, desc="Converting"),
                                    verbose=args.verbose):
            convert_passage(passage, report_writer=writer)
            write_passage(passage, outdir=args.outdir, prefix=args.prefix, verbose=args.verbose)
            f.flush()
    print("Wrote '%s'" % args.outfile)


if __name__ == "__main__":
    argparser = ArgumentParser(description=desc)
    argparser.add_argument("passages", nargs="+", help="the corpus, given as xml/pickle file names")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-O", "--outfile", default=os.path.splitext(argparser.prog)[0] + ".csv", help="log file")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
