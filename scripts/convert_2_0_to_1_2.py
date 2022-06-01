import csv
import os
from argparse import ArgumentParser

from ucca import layer1
from ucca.ioutil import get_passages_with_progress_bar, write_passage
from ucca.normalization import destroy, copy_edge

desc = """Convert the English Wiki corpus from version 2.0 to 1.2"""


def replace_time_and_quantifier(edge):
    if edge.tag in (layer1.EdgeTags.Time, layer1.EdgeTags.Quantifier):
        edge.tag = layer1.EdgeTags.Adverbial if edge.parent.is_scene() else layer1.EdgeTags.Elaborator
        if len(edge.parent.parents) == 1 and edge.parent.incoming[0].tag == edge.tag:
            for e in edge.parent:
                copy_edge(e, parent=edge.parent.parents[0])
            destroy(edge.parent)
        return True
    return False


RULES = (replace_time_and_quantifier,)


def convert_passage(passage, report_writer):
    for rule in RULES:
        for node in passage.layer(layer1.LAYER_ID).all:
            for edge in node:
                parent = edge.parent
                parent_str = str(parent)
                if rule(edge):
                    report_writer.writerow((rule.__name__, passage.ID, edge, parent_str, parent))


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("rule", "passage", "edge", "before", "after"))
        for passage in get_passages_with_progress_bar(args.passages, desc="Converting"):
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
    argparser.add_argument("-v", "--verbose", action="store_true", help="print more information")
    main(argparser.parse_args())
