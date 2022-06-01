from collections import OrderedDict

from argparse import ArgumentParser

from ucca.constructions import extract_candidates, add_argument
from ucca.ioutil import get_passages_with_progress_bar, external_write_mode


def main(args):
    for passage in get_passages_with_progress_bar(args.passages):
        c2es = OrderedDict((c, [candidate.edge for candidate in candidates]) for c, candidates in
                           extract_candidates(passage, constructions=args.constructions, verbose=args.verbose).items()
                           if candidates)
        if any(c2es.values()):
            with external_write_mode():
                if not args.verbose:
                    print("%s:" % passage.ID)
                for construction, edges in c2es.items():
                    if edges:
                        print("  %s:" % construction.description)
                        for edge in edges:
                            print("    %s [%s %s]" % (edge, edge.tag, edge.child))
                print()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Extract linguistic constructions from UCCA corpus.")
    argparser.add_argument("passages", nargs="+", help="the corpus, given as xml/pickle file names")
    add_argument(argparser, False)
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
