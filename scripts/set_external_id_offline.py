#!/usr/bin/env python3
import argparse
import os
import sys

from ucca.ioutil import get_passages_with_progress_bar, write_passage

desc = """Rename passages by a given mapping of IDs"""


def main(filename, input_filenames, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(filename, encoding="utf-8") as f:
        pairs = [line.strip().split() for line in f]
        old_to_new_id = {old_id: new_id for new_id, old_id in pairs}
    for passage in get_passages_with_progress_bar(input_filenames, desc="Renaming"):
        passage._ID = old_to_new_id[passage.ID]
        write_passage(passage, outdir=outdir, verbose=False)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=desc)
    argument_parser.add_argument("filename", help="file with lines of the form <NEW ID> <OLD ID>")
    argument_parser.add_argument("input_filenames", help="filename pattern or directory with input passages")
    argument_parser.add_argument("-o", "--outdir", default=".", help="output directory")
    main(**vars(argument_parser.parse_args()))
    sys.exit(0)
