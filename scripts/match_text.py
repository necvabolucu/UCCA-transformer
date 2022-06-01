import argparse
import re
import sys
from glob import glob
from itertools import groupby
from operator import attrgetter

from tqdm import tqdm

from ucca import layer0
from ucca.ioutil import get_passages_with_progress_bar


def gen_lines(filenames):
    for filename in glob(filenames) or [filenames]:
        with open(filename, encoding="utf-8") as f:
            try:
                for line in map(str.strip, f):
                    if line and not line.startswith("#"):
                        yield re.sub(r"\[\d+\]", "", line)  # Remove numbers inside brackets
            except UnicodeDecodeError as e:
                raise IOError("Failed reading '%s'" % filename) from e


class CandidateMatcher:
    def __init__(self, text):
        self.text = text
        self.char_map = {}
        no_space_chars = []
        for i, char in enumerate(text):
            if not char.isspace():
                self.char_map[len(no_space_chars)] = i
                no_space_chars.append(char)
        self.no_space_text = "".join(no_space_chars)

    def __call__(self, no_space_text):
        try:
            index = self.no_space_text.index(no_space_text)
            return self.text[self.char_map[index]:self.char_map[index + len(no_space_text) - 1] + 1]
        except ValueError:
            return None


def match_passage_text(passage, matchers, out):
    passage_tokens = sorted(passage.layer(layer0.LAYER_ID).all, key=attrgetter("position"))
    for paragraph, terminals in groupby(passage_tokens, key=attrgetter("paragraph")):
        tokens = [terminal.text for terminal in terminals]
        no_space_text = "".join(tokens)
        match = next(filter(None, (matcher(no_space_text) for matcher in matchers)), "@@@" + " ".join(tokens))
        print(passage.ID, match, sep="\t", file=out)


def alternative_spellings(text):
    yield text


def main(args):
    matchers = [CandidateMatcher(spelling) for line in tqdm(list(gen_lines(args.text)),
                                                            desc="Indexing " + args.text, unit=" lines")
                for spelling in alternative_spellings(line)]
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for p in get_passages_with_progress_bar(args.filenames, desc="Matching", converters={}):
        match_passage_text(p, matchers, out)
    out.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Match UCCA passages to original text and print aligned lines")
    argparser.add_argument("text", help="file of text to match to")
    argparser.add_argument("filenames", nargs="+", help="files or directories of UCCA passages to match")
    argparser.add_argument("-o", "--out", default="text.tsv", help="output file")
    argparser.add_argument("-l", "--lang", default="en", help="spaCy language")
    main(argparser.parse_args())
