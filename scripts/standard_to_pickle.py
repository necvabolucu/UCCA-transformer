#!/usr/bin/env python3
import sys

import argparse
import os
from tqdm import tqdm

from ucca.ioutil import file2passage, passage2file, external_write_mode

desc = """Parses an XML in UCCA standard format, and writes them in binary Pickle format."""


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    for filename in tqdm(args.filenames, desc="Converting", unit=" passages"):
        if args.verbose:
            with external_write_mode():
                print("Reading passage '%s'..." % filename, file=sys.stderr)
        passage = file2passage(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        outfile = args.outdir + os.path.sep + basename + ".pickle"
        if args.verbose:
            with external_write_mode():
                print("Writing file '%s'..." % outfile, file=sys.stderr)
        passage2file(passage, outfile, binary=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('filenames', nargs='+', help="XML file names to convert")
    argparser.add_argument('-o', '--outdir', default='.', help="output directory")
    argparser.add_argument('-v', '--verbose', action="store_true", help="verbose output")
    main(argparser.parse_args())
