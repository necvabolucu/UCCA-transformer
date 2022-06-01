#! /usr/bin/python3

import argparse
import pickle
from xml.etree.ElementTree import ElementTree, fromstring

import psycopg2

import ucca.convert

desc = """Parses an XML in UCCA site format.

The input can be given as either an XML file or a DB file with passage ID
and user name, and the output is either the standard format XML or
a pickled object.
Possible input methods are using a DB file with pid and user, which gets the
annotation of the specified user for the specified passage from teh DB file,
or using filename of a site-formatted XML file.

"""


def site2passage(filename):
    """Opens a file and returns its parsed Passage object"""
    with open(filename, encoding="utf-8") as f:
        etree = ElementTree().parse(f)
    return ucca.convert.from_site(etree)


def db2passage(handle, pid, user):
    """Gets the annotation of user to pid from the DB handle - returns a passage"""
    handle.execute("SET search_path to oabend")
    handle.execute("SELECT id FROM users WHERE username=%s", (user,))
    uid = handle.fetchone()[0]
    handle.execute("SELECT xml,ts FROM xmls WHERE paid=%s AND uid=%s " +
                   "ORDER BY ts DESC", (pid, uid))
    raw_xml, ts = handle.fetchone()
    #print('extracted passage from '+str(ts))
    return ucca.convert.from_site(fromstring(raw_xml))


def main(args):
    # Checking for illegal combinations
    if args.db and args.filename:
        argparser.error("Only one source, XML or DB file, can be used")
    if (not args.db) and (not args.filename):
        argparser.error("Must specify one source, XML or DB file")
    if args.db and not (args.pid and args.user):
        argparser.error("Must specify a username and a passage ID when " +
                        "using DB file option")
    if (args.pid or args.user) and not args.db:
        argparser.error("Cannot use user and passage ID options without DB file")

    if args.filename:
        passage = site2passage(args.filename)
    else:
        conn = psycopg2.connect(host=args.host, database=args.db)
        c = conn.cursor()
        passage = db2passage(c, args.pid, args.user)

    if args.binary:
        with open(args.binary, "wb") as binf:
            pickle.dump(passage, binf)
    else:
        output = ucca.convert.to_text(passage, lang=args.lang)
        if args.outfile:
            with open(args.outfile, "w", encoding="utf-8") as outf:
                outf.write(output)
        else:
            print(output)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filename", nargs="?", help="XML file name to convert")
    argparser.add_argument("-o", "--outfile", help="output file for standard XML")
    argparser.add_argument("-b", "--binary", help="output file for binary pickel")
    argparser.add_argument("-d", "--db", help="DB file to get input from")
    argparser.add_argument("--host", help="DB host server to get input from")
    argparser.add_argument("-p", "--pid", type=int, help="PassageID to query DB")
    argparser.add_argument("-u", "--user", help="Username to DB query")
    argparser.add_argument("-l", "--lang", default="en", help="language two-letter code for sentence model")
    main(argparser.parse_args())
