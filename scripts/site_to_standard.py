import argparse
import os
import sqlite3
from glob import glob
from xml.etree.ElementTree import ElementTree, fromstring

import ucca.convert
from ucca.ioutil import write_passage

desc = """Parses an XML in UCCA site format.

The input can be given as either an XML file or a DB file with passage ID
and user name, and the output is either the standard format XML or
a pickled object.
Possible input methods are using a DB file with pid and user, which gets the
annotation of the specified user for the specified passage from teh DB file,
or using filenames of a site-formatted XML file.

"""


def site2passage(filename):
    """Opens a file and returns its parsed Passage object"""
    with open(filename, encoding="utf-8") as f:
        print("Reading '%s'..." % filename)
        return ucca.convert.from_site(ElementTree().parse(f))


def db2passage(handle, pid, user):
    """Gets the annotation of user to pid from the DB handle - returns a passage"""
    handle.execute("SELECT id FROM users WHERE username=?", (user,))
    uid = handle.fetchone()[0]
    handle.execute("SELECT xml FROM xmls WHERE paid=? AND uid=? ORDER BY ts DESC", (pid, uid))
    return ucca.convert.from_site(fromstring(handle.fetchone()[0]))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for filename, passage in ((filename, site2passage(filename)) for pattern in args.filenames
                              for filename in sorted(glob(pattern)) or [pattern]) if args.filenames \
            else ((pid, db2passage(sqlite3.connect(args.db).cursor(), pid, args.user)) for pid in args.pids):
        write_passage(passage, outdir=args.out_dir, binary=args.binary)


def check_illegal_combinations(args):
    if args.db and not (args.pids and args.user):
        argparser.error("Must specify a username and a passage ID when using DB file option")
    if (args.pids or args.user) and not args.db:
        argparser.error("Cannot use user and passage ID options without DB file")
    return args


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="*", help="XML file name to convert")
    argparser.add_argument("-d", "--db", help="DB file to get input from")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory for standard XML")
    argparser.add_argument("-b", "--binary", help="output file for binary pickle")
    argparser.add_argument("-p", "--pids", nargs="*", type=int, help="PassageIDs to query DB")
    argparser.add_argument("-u", "--user", help="Username to DB query")
    main(check_illegal_combinations(argparser.parse_args()))
