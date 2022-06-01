import argparse
import os
from glob import glob

desc = """Replaces the tokens according to a dictionary."""


def read_dictionary_from_file(filename):
    f = open(filename, encoding="utf-8")
    d = {}
    for line in f:
        fields = line.strip().split()
        d[fields[0]] = fields[1]
        d[fields[0].strip().encode('ascii', 'xmlcharrefreplace').decode()] = \
            fields[1].strip().encode('ascii', 'xmlcharrefreplace').decode()
    print(d)
    return d


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    replacement_dict = read_dictionary_from_file(args.dict)
    for pattern in args.filenames:
        for filename in sorted(glob(pattern)) or [pattern]:
            basename = os.path.basename(filename)
            with open(os.path.join(args.out_dir, basename), "w", encoding="utf-8") as outfile:
                with open(filename, encoding="utf-8") as infile:
                    xml_string = infile.read()
                for k, v in replacement_dict.items():
                    if args.whole_word:
                        xml_string = xml_string.replace("text=\"" + k + "\"", "text=\"" + v + "\"")
                    else:
                        xml_string = xml_string.replace(k, v)
                print(xml_string, file=outfile, end="")
    print("Done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="files to replace tokens in")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory for changed XMLs")
    argparser.add_argument("-d", "--dict",
                           help="filename to read the dictionary from. the file should have one line per entry, in the"
                                " format of <original text> <replaced text>")
    argparser.add_argument("-w", "--whole-word", action="store_true", help="replace whole word")
    main(argparser.parse_args())
