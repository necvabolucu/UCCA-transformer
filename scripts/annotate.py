#!/usr/bin/env python3

import argparse

from ucca.ioutil import write_passage, get_passages_with_progress_bar
from ucca.textutil import annotate_all, is_annotated

desc = """Read UCCA standard format in XML or binary pickle, and write back with POS tags and dependency parse."""


def main(args):
    for passage in annotate_all(get_passages_with_progress_bar(args.filenames, desc="Annotating"),
                                replace=True, as_array=args.as_array, verbose=args.verbose):
        assert is_annotated(passage, args.as_array), "Passage %s is not annotated" % passage.ID
        write_passage(passage, outdir=args.out_dir, verbose=args.verbose)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to annotate")
    argparser.add_argument("-o", "--out-dir", default=".", help="directory to write annotated files to")
    argparser.add_argument("-a", "--as-array", action="store_true", help="save annotations as array in passage level")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
