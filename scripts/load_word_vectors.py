#!/usr/bin/env python3

import argparse

from ucca.textutil import get_word_vectors

desc = """Load word vectors file to make sure it works."""


def main(args):
    for filename in args.filenames:
        vectors, dim = get_word_vectors(size=args.rows, dim=args.dim, filename=filename)
        print("Loaded %d rows, dim=%d" % (len(vectors), dim))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="word vector files to load")
    argparser.add_argument("-r", "--rows", type=int, help="maximum number of word vectors")
    argparser.add_argument("-d", "--dim", type=int, help="maximum dimension of word vectors")
    main(argparser.parse_args())
