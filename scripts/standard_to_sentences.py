#!/usr/bin/env python3

import argparse
import os
import sys
from itertools import count
from logging import warning

from ucca.convert import split2sentences, split_passage
from ucca.ioutil import passage2file, get_passages_with_progress_bar, external_write_mode
from ucca.normalization import normalize
from ucca.textutil import extract_terminals

desc = """Parses XML files in UCCA standard format, and writes a passage per sentence."""

NUM_NODES_WARNING = 500  # Warn if a sentence has more than this many nodes


class Splitter:
    def __init__(self, sentences, enum=False, suffix_format="%03d", suffix_start=0):
        self.sentences = sentences
        self.sentence_to_index = {}
        for i, sentence in enumerate(sentences):
            self.sentence_to_index.setdefault(sentence, []).append(i)
        self.enumerate = enum
        self.suffix_format = suffix_format
        self.suffix_start = suffix_start
        self.index = 0
        self.matched_indices = set()

    @classmethod
    def read_file(cls, filename, **kwargs):
        if filename is None:
            return None
        with open(filename, encoding="utf-8") as f:
            sentences = [line.strip() for line in f]
        return cls(sentences, **kwargs)

    def split(self, passage):
        ends = []
        ids = []
        token_lists = []
        for terminal in extract_terminals(passage):
            token_lists.append([])
            for terminals in token_lists if self.index is None else [token_lists[0]]:
                terminals.append(terminal)
                sentence = " ".join(t.text for t in terminals)
                if self.index is not None and self.index < len(self.sentences) and self.sentences[
                        self.index].startswith(sentence):  # Try matching next sentence rather than shortest
                    index = self.index if self.sentences[self.index] == sentence else None
                else:
                    indices = self.sentence_to_index.get(sentence)
                    index = self.index = indices.pop(0) if indices else None
                if index is not None:
                    self.matched_indices.add(index)
                    last_end = terminals[0].position - 1
                    if len(terminals) > 1 and last_end and last_end not in ends:
                        ends.append(last_end)
                    ends.append(terminal.position)
                    ids.append(str(index))
                    token_lists = []
                    self.index += 1
                    break
        return split_passage(passage, ends, ids=ids if self.enumerate else None,
                             suffix_format=self.suffix_format, suffix_start=self.suffix_start)


def main(args):
    splitter = Splitter.read_file(args.sentences, enum=args.enumerate,
                                  suffix_format=args.suffix_format, suffix_start=args.suffix_start)
    os.makedirs(args.outdir, exist_ok=True)
    i = 0
    for passage in get_passages_with_progress_bar(args.filenames, "Splitting"):
        for sentence in splitter.split(passage) if splitter else split2sentences(
                passage, remarks=args.remarks, lang=args.lang, ids=map(str, count(i)) if args.enumerate else None):
            i += 1
            outfile = os.path.join(args.outdir, args.prefix + sentence.ID + (".pickle" if args.binary else ".xml"))
            if len(sentence.nodes) > NUM_NODES_WARNING:
                warning(f"Sentence {i} in passage {passage.ID} has {len(sentence.nodes)} > {NUM_NODES_WARNING} nodes")
            if args.verbose:
                with external_write_mode():
                    print(sentence, file=sys.stderr)
                    print("Writing passage file for sentence '%s'..." % outfile, file=sys.stderr)
            if args.normalize:
                normalize(sentence)
            passage2file(sentence, outfile, binary=args.binary)
    if splitter and len(splitter.matched_indices) < len(splitter.sentences):
        print("", "Unmatched sentences:", *[s for i, s in enumerate(splitter.sentences)
                                            if i not in splitter.matched_indices], sep="\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to convert")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-f", "--suffix-format", default="%03d", help="sentence number suffix format")
    argparser.add_argument("-i", "--suffix-start", type=int, default=0, help="start index for number suffix")
    argparser.add_argument("-r", "--remarks", action="store_true", help="annotate original IDs")
    argparser.add_argument("-l", "--lang", default="en", help="language two-letter code for sentence model")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-s", "--sentences", help="optional input file with sentence at each line to split by")
    argparser.add_argument("-e", "--enumerate", action="store_true", help="set each output sentence ID by global order")
    argparser.add_argument("-N", "--no-normalize", dest="normalize", action="store_false",
                           help="do not normalize passages after splitting")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print information about every split sentence")
    main(argparser.parse_args())
