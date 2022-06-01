#!/usr/bin/env python3

import argparse
import os
import sys
from itertools import count

from ucca.convert import split2paragraphs
from ucca.ioutil import passage2file, get_passages_with_progress_bar, external_write_mode
from ucca.normalization import normalize

desc = """Parses XML files in UCCA standard format, and writes a passage per paragraph."""


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    i = 0
    for passage in get_passages_with_progress_bar(args.filenames, "Splitting"):
        for paragraph in split2paragraphs(
                passage, remarks=args.remarks, lang=args.lang, ids=map(str, count(i)) if args.enumerate else None):
            i += 1
            outfile = os.path.join(args.outdir, args.prefix + paragraph.ID + (".pickle" if args.binary else ".xml"))
            if args.verbose:
                with external_write_mode():
                    print(paragraph, file=sys.stderr)
                    print("Writing passage file for paragraph '%s'..." % outfile, file=sys.stderr)
            if args.normalize:
                normalize(paragraph)
            passage2file(paragraph, outfile, binary=args.binary)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to convert")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-r", "--remarks", action="store_true", help="annotate original IDs")
    argparser.add_argument("-l", "--lang", default="en", help="language two-letter code for paragraph model")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-e", "--enumerate", action="store_true", help="set output paragraph ID by global order")
    argparser.add_argument("-N", "--no-normalize", dest="normalize", action="store_false",
                           help="do not normalize passages after splitting")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print information about every split paragraph")
    main(argparser.parse_args())
