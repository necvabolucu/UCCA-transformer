import argparse
import os
import pickle
from glob import glob
from xml.etree.ElementTree import Element

import ucca.convert
from ucca.ioutil import write_passage

desc = """Parses pickle files containing XML in UCCA site format, and convert to standard XML"""


def pickle_site2passage(filename):
    """Opens a pickle file containing XML in UCCA site format and returns its parsed Passage object"""
    with open(filename, "rb") as h:
        root = elem = pickle.load(h)
        while isinstance(elem, list):
            try:
                elem = next(e for e in elem if isinstance(e, (Element, list)))
            except StopIteration:
                raise ValueError("Cannot parse %s" % root)
        return ucca.convert.from_site(elem)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    exceptions = []
    for pattern in args.filenames:
        for filename in sorted(glob(pattern)) or [pattern]:
            print("Reading '%s'..." % filename)
            try:
                passage = pickle_site2passage(filename)
                write_passage(passage, outdir=args.out_dir, binary=args.binary, basename=os.path.basename(filename))
            except ValueError as e:
                exceptions.append((filename, e))
    if exceptions:
        for filename, e in exceptions:
            print("'%s': %s" % (filename, e))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="*", help="pickle file names to convert")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-b", "--binary", help="output binary pickle")
    main(argparser.parse_args())
