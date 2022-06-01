import argparse
import os

from ucca.ioutil import get_passages_with_progress_bar, write_passage
from ucca.normalization import normalize


def main(args):
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    for p in get_passages_with_progress_bar(args.filenames, desc="Normalizing", converters={}):
        normalize(p, extra=args.extra)
        write_passage(p, outdir=args.outdir, prefix=args.prefix, binary=args.binary, verbose=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Normalize UCCA passages")
    argparser.add_argument("filenames", nargs="+", help="files or directories to normalize")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-e", "--extra", action="store_true", help="extra normalization rules")
    main(argparser.parse_args())
