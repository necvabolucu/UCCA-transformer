#!/usr/bin/env python3
"""
The evaluation software for UCCA layer 1.
"""

from argparse import ArgumentParser

from ucca import convert, constructions
from ucca.evaluation import evaluate
from ucca_db import api


def main(args):
    keys = [args.guessed, args.ref]
    xmls = api.get_by_xids(db_name=args.db_filename, host_name=args.host, xids=keys) if args.from_xids else \
        api.get_xml_trees(db_name=args.db_filename, host_name=args.host, pid=args.pid, usernames=keys)
    guessed, ref = [convert.from_site(x) for x in xmls]
    if args.units or args.fscore or args.errors:
        evaluate(guessed, ref, units=args.units, fscore=args.fscore, errors=args.errors,
                 constructions=args.constructions, verbose=True)


if __name__ == '__main__':
    argparser = ArgumentParser(description="Evaluate passages on UCCA DB")
    argparser.add_argument("--db", "-d", required=True, dest="db_filename", help="the db file name")
    argparser.add_argument("--host", "--hst", help="the host name")
    group = argparser.add_mutually_exclusive_group()
    group.add_argument("-p", "--pid", type=int, help="the passage ID")
    group.add_argument("-x", "--from_xids", action="store_true",
                       help="interpret the ref and the guessed parameters as Xids in the db")
    argparser.add_argument("--guessed", "-g", required=True,
                           help="if a db is defined - the username for the guessed annotation; "
                                "else - the xml file name for the guessed annotation")
    argparser.add_argument("-r", "--ref", required=True,
                           help="if a db is defined - the username for the reference annotation; "
                                "else - the xml file name for the reference annotation")
    argparser.add_argument("-u", "--units", action="store_true",
                           help="the units the annotations have in common, and those each has separately")
    argparser.add_argument("-f", "--fscore", action="store_true",
                           help="outputs the traditional P,R,F instead of the scene structure evaluation")
    argparser.add_argument("-e", "--errors", action="store_true",
                           help="prints the error distribution according to its frequency")
    constructions.add_argument(argparser)
    main(argparser.parse_args())
