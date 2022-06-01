#!/usr/bin/env python3

import argparse
import csv
import os
from xml.etree.ElementTree import Element

from unidecode import unidecode

from ucca.convert import to_site, from_site, SiteCfg, SiteUtil
from ucca.ioutil import get_passages_with_progress_bar, write_passage
from ucca.normalization import normalize
from ucca.textutil import get_tokenizer

desc = """Parses XML files in UCCA standard format, fix tokenization and write back."""

PARENT_TYPE = "org"
CONTEXT = 6
CURRENCIES = {"$", "¥", "£", "€"}

# ---------------------------- set format:
HYPHEN_TO_E_AND_C = "elaborator-"
APOSTROPHE_TO_E_AND_C = "elaborator'"
HYPHEN_TO_UNANALYZABLE = "unanalyzable-"
APOSTROPHE_TO_UNANALAYZABLE = "unanalyzable'"
POSSESSIVE_S_TO_UNANALAYZABLE = "possessive s - unanalyzable"
POSSESSIVE_S_TO_PARENT_AND_F = "possessive s - Function"
POSSESSIVE_S_TO_C_AND_R = "possessive s - Relator"


def expand_to_neighboring_punct(i, is_puncts):
    """
    >>> expand_to_neighboring_punct(0, [False, True, True])
    (0, 3)
    >>> expand_to_neighboring_punct(2, [True, True, False])
    (0, 3)
    >>> expand_to_neighboring_punct(1, [False, False, False])
    (1, 2)
    """
    start = i
    end = i + 1
    while start > 0 and is_puncts[start - 1]:
        start -= 1
    while end < len(is_puncts) and is_puncts[end]:
        end += 1
    return start, end


class State:
    def __init__(self):
        self.ID = 1000000

    def get_id(self):
        ret = str(self.ID)
        self.ID += 1
        return ret


def create_unit_element(state, text, tag):
    elem = Element(SiteCfg.Tags.Terminal,
                   {SiteCfg.Attr.SiteID: state.get_id()})
    elem.text = text
    preterminal_elem = Element(SiteCfg.Tags.Unit,
                               {SiteCfg.Attr.SiteID: state.get_id()})
    preterminal_parent = Element(SiteCfg.Tags.Unit,
                                 {
                                     SiteCfg.Attr.ElemTag: tag,
                                     SiteCfg.Attr.SiteID: state.get_id(),
                                     SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                                     SiteCfg.Attr.Uncertain: SiteCfg.FALSE})
    preterminal_elem.append(elem)
    preterminal_parent.append(preterminal_elem)
    return preterminal_parent


def create_token_element(state, text, is_punctuation):
    elem = Element(SiteCfg.Tags.Terminal, {SiteCfg.Attr.SiteID: state.get_id()})
    elem.text = text
    preterminal_elem = Element(SiteCfg.Tags.Unit,
                               {SiteCfg.Attr.ElemTag: SiteCfg.Types.Punct if is_punctuation else SiteCfg.TBD,
                                SiteCfg.Attr.SiteID: state.get_id(),
                                SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                                SiteCfg.Attr.Uncertain: SiteCfg.FALSE})
    preterminal_elem.append(elem)
    return preterminal_elem


def insert_punct(insert_index, preterminal_parent, state, punct_tokens):
    for punct_token in punct_tokens:
        punct_elem = create_token_element(state, punct_token, is_punctuation=True)
        preterminal_parent.insert(insert_index, punct_elem)
        insert_index += 1
    return insert_index


def get_parents(paragraph, elements):
    return [next(x for x in paragraph.iter(SiteCfg.Tags.Unit) if t in x) for t in elements]


def insert_retokenized(terminal, preterminal_parent, tokens, index_in_preterminal_parent, non_punct_index, state):
    terminal.text = tokens[non_punct_index]
    insert_index = insert_punct(index_in_preterminal_parent, preterminal_parent, state, tokens[:non_punct_index])
    insert_punct(insert_index + 1, preterminal_parent, state, tokens[non_punct_index + 1:])


def insert_retokenized_currency(i, terminals, preterminals,
                                preterminal_parents,
                                tokens, state):
    if len(tokens) == 2 and tokens[0] in CURRENCIES and \
            (tokens[1].replace('.', '', 1).isdigit() or
             tokens[1].replace(',', '').isdigit()):
        terminals[i].text = tokens[1]
        index_to_insert = preterminal_parents[i].getchildren(). \
            index(preterminals[i])
        preterminal_parents[i].insert(index_to_insert,
                                      create_token_element(state,
                                                           tokens[0],
                                                           is_punctuation=False))
        return True
    return False


def false_indices(l):
    return [i for i, x in enumerate(l) if not x]


def is_punct(text):
    return all(not c.isalnum() for c in text)


def strip_context(new_context, old_context, start_offset, end_offset):
    """
    >>> strip_context(["I", "'ve", "done"], ["I", "'ve", "done"], 1, 1)
    ["'ve"]
    >>> strip_context(["I", "'ve,", "done"], ["I", "'ve", ",", "done"], 1, 1)
    ["'ve", ","]
    >>> strip_context(["'ve", "done"], ["'", "ve", "done"], 0, 1)
    ["'", "ve"]
    >>> strip_context(["I", "'ve,"], ["I", "'ve", ","], 1, 0)
    ["'ve", ","]
    >>> strip_context(["can", "'t", "see"], ["ca", "n't", "see"], 1, 1)
    ["'t"]
    >>> strip_context(["I", "can", "'t"], ["I", "ca", "n't"], 1, 1)
    ["can"]
    >>> strip_context(["because", "somebody", "'d"], ["because", "somebody'd"], 1, 1)
    []
    >>> strip_context(["somebody", "'d", "always"], ["somebody'd", "always"], 1, 1)
    []
    """
    start = 0
    if start_offset:
        prefix = ""
        while old_context[0].startswith(prefix + new_context[start]):
            prefix += new_context[start]
            start += 1
        diff = len(old_context[0]) - len(prefix)
        if diff:
            if start > 0:
                new_context[start - 1] += new_context[start][:diff]
            new_context[start] = new_context[start][diff:]
    end = len(new_context)
    if end_offset:
        suffix = ""
        while old_context[-1].endswith(new_context[end - 1] + suffix):
            suffix = new_context[end - 1] + suffix
            end -= 1
        diff = len(old_context[-1]) - len(suffix)
        if diff:
            end -= 1
            if end > 0:
                new_context[end - 1] += new_context[end][:-diff]
            new_context[end] = new_context[end][-diff:]
    return new_context[start:end]


END_CLITICS = {"'m", "'ll", "'s", "'ve", "'d", "'re", "n't", "'t"}
START_CLITICS = {"l'", "qu'", "n'", "d'", "s'", "m'", "c'", "t'", "jusqu'", "j'"}


def insert_spaces(tokens):
    for token, next_token in zip(tokens[:-1], tokens[1:]):
        yield token
        if token.lower() not in START_CLITICS and next_token.lower() not in END_CLITICS\
                and not (token == next_token == "'"):
            yield " "
    if tokens:
        yield tokens[-1]


# ------------------------- split cases ----------------------------

def split_possessive_s_unanalyzable(i, terminals, preterminals,
                                    preterminal_parents, state):
    """split possessive s as unanalyzable. xxx's -> xxx 's.
    use when the original token is unanalyzable."""
    without = terminals[i].text.strip("'s")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    preterminal_parents[i].insert(index_to_insert,
                                  create_token_element(state, without,
                                                       is_punctuation=False))
    terminals[i].text = "'s"


def split_apostrophe_unanalyzable(i, terminals, preterminals,
                                  preterminal_parents, state):
    """Split apostrophe as unanalyzable. x'xxx -> x' xxx.
    use when the original token is unanalyzable."""
    split_list = terminals[i].text.split("'")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    preterminal_parents[i].insert(index_to_insert,
                                  create_token_element(state, split_list[0] + "'",
                                                       is_punctuation=False))
    terminals[i].text = split_list[1]


def split_hyphen_unanalyzable(i, terminals, preterminals,
                              preterminal_parents, state):
    """split token with hyphens to unanalyzable tokens. xxx-xxx-xx ->
    xxx - xxx - xx"""
    divided = terminals[i].text.split("-")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    words = divided[1:]
    counter = 1
    preterminal_parents[i].insert(index_to_insert,
                                  create_token_element(state, divided[0],
                                                       is_punctuation=False))
    for word in words:
        preterminal_parents[i].insert(index_to_insert + counter,
                                      create_token_element(state, "-", is_punctuation=True))
        preterminal_parents[i].insert(index_to_insert + counter + 1,
                                      create_token_element(state, word, is_punctuation=False))
        counter += 2
    preterminal_parents[i].remove(preterminals[i])


def split_apostrophe_to_units(i, terminals, preterminals, preterminal_parents,
                              tag1, tag2, state):
    """split token with apostrophe to Elaborator and Center. x'xxx -> x' xxx"""
    divided = terminals[i].text.split("'")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, divided[1],
                                                      tag2))
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, divided[0] + "'",
                                                      tag1))
    preterminal_parents[i].remove(preterminals[i])


def split_hyphen_to_units(i, terminals, preterminals, preterminal_parents, tag1, tag2, state):
    """split token with hyphen to two different units. xxx-xxx -> xxx - xxx"""
    divided = terminals[i].text.split("-")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, divided[1],
                                                      tag2))
    preterminal_parents[i].insert(index_to_insert,
                                  create_token_element(state, "-", is_punctuation=True))
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, divided[0],
                                                      tag1))
    preterminal_parents[i].remove(preterminals[i])


def split_possessive_s_to_units(i, terminals, preterminals,
                                preterminal_parents, state, tag1, tag2):
    """split possessive s to two different units. xxx's -> xxx 's"""
    without = terminals[i].text.strip("'s")
    index_to_insert = preterminal_parents[
        i].getchildren().index(preterminals[i])
    first_type = preterminal_parents[i].get(SiteCfg.Attr.ElemTag) if tag1 == PARENT_TYPE \
        else tag1
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, "'s",
                                                      tag2))
    preterminal_parents[i].insert(index_to_insert,
                                  create_unit_element(state, without,
                                                      first_type))
    preterminal_parents[i].remove(preterminals[i])


# --------------------------------------------------------------------


def context(i, terminals):
    start = max(i - CONTEXT, 0)
    end = min(len(terminals) - 1, i + CONTEXT)
    context_tokens = []
    for elem in terminals[start:end + 1]:
        context_tokens.append(elem.text)
    return context_tokens


def handle_words_set(rule, i, terminals, preterminals, preterminal_parents,
                     state):
    """use set of words to determine the right fix needed"""
    if rule == HYPHEN_TO_UNANALYZABLE:
        split_hyphen_unanalyzable(i, terminals, preterminals,
                                  preterminal_parents, state)
        return True
    if rule == POSSESSIVE_S_TO_UNANALAYZABLE:
        split_possessive_s_unanalyzable(i, terminals, preterminals,
                                        preterminal_parents, state)
        return True
    if rule == APOSTROPHE_TO_UNANALAYZABLE:
        split_apostrophe_unanalyzable(i, terminals, preterminals,
                                      preterminal_parents, state)
        return True
    if preterminal_parents[i].get(SiteCfg.Attr.Unanalyzable) == SiteCfg.TRUE:
        return False  # if token is unanalyzable, do nothing of the next steps.
    if rule == HYPHEN_TO_E_AND_C:
        split_hyphen_to_units(i, terminals, preterminals, preterminal_parents,
                              "Elaborator", "Center", state)
        return True
    if rule == APOSTROPHE_TO_E_AND_C:
        split_apostrophe_to_units(i, terminals, preterminals,
                                  preterminal_parents,
                                  "Elaborator", "Center", state)
        return True
    if rule == POSSESSIVE_S_TO_PARENT_AND_F:
        split_possessive_s_to_units(i, terminals, preterminals,
                                    preterminal_parents, state, PARENT_TYPE,
                                    "Function")
        return True
    if rule == POSSESSIVE_S_TO_C_AND_R:
        split_possessive_s_to_units(i, terminals, preterminals,
                                    preterminal_parents, state,
                                    "Center", "Relator")
        return True
    return False


def retokenize(i, start, end, terminals, preterminals, preterminal_parents,
               passage_id, tokenizer, state, cw, words):
    start_offset = 0 if start == 0 else 1
    end_offset = 0 if end == len(terminals) else 1
    old_context = [s for t in terminals[start - start_offset:end + end_offset]
                   for s in SiteUtil.unescape(
            t.text).strip().split()]  # In case a token happens to contain a space
    new_context = [t.orth_ for t in
                   tokenizer("".join(insert_spaces(old_context)))]
    if old_context == new_context:
        return False
    old_tokens = old_context[start_offset:len(old_context) - end_offset]
    new_tokens = strip_context(new_context, old_context, start_offset,
                               end_offset)
    if not new_tokens or old_tokens == new_tokens:
        return False
    non_punct_indices = false_indices(map(is_punct, new_tokens))
    to_write = fixed = None
    if words is not None and terminals[i].text in words:
        if handle_words_set(words[terminals[i].text], i, terminals, preterminals,
                            preterminal_parents, state):
            fixed = True
            to_write = "Fixed - Set"
        else:
            to_write = "Unhandled - bad set"
    elif "-" in terminals[i].text:
        without = terminals[i].text.split("-")
        if all(word[0].isupper() and word.isalnum() for word in without):
            split_hyphen_unanalyzable(i, terminals,
                                      preterminals, preterminal_parents,
                                      state)
            fixed = True
            to_write = "Fixed - Names"
    elif len(non_punct_indices) == 1:  # Only one token in the sequence is not punctuation
        non_punct_index = non_punct_indices[0]
        new_tokens = (decode_special_chars(new_tokens[:non_punct_index]) + [
            new_tokens[non_punct_index]] +
                      decode_special_chars(new_tokens[
                                           non_punct_index + 1:]))  # Replace special charas in punct
        index_in_preterminal_parent = preterminal_parents[
            i].getchildren().index(preterminals[i])
        if insert_retokenized_currency(i, terminals, preterminals,
                                       preterminal_parents, new_tokens, state):
            to_write = "Fixed - currency"
        else:
            for j in list(range(start, i)) + list(
                    range(i + 1, end)):  # Remove all surrounding punct
                preterminal_parents[j].remove(preterminals[j])
            insert_retokenized(terminals[i], preterminal_parents[i],
                               new_tokens, index_in_preterminal_parent,
                               non_punct_index, state)
            to_write = "Fixed"
        fixed = True
    cw.writerow(
        (to_write, passage_id, " ".join(old_tokens), " ".join(new_tokens),
         " ".join(context(i, terminals))))
    return fixed


def decode_special_chars(tokens):  # Replace special chars with ascii variants but only if length is preserved
    return [d if len(t) == len(d) or all(c == "." for c in d) else t
            for t, d in zip(tokens, (unidecode(t) for t in tokens))]


def fix_tokenization(passage, words_set, lang, cw):
    tokenizer = get_tokenizer(lang=lang)
    elem = to_site(passage)
    state = State()
    ever_changed = False
    for paragraph in elem.iterfind(SiteCfg.Paths.Paragraphs):
        while True:
            changed = False
            terminals = list(paragraph.iter(SiteCfg.Tags.Terminal))
            preterminals = get_parents(paragraph, terminals)
            preterminal_parents = get_parents(paragraph, preterminals)
            is_puncts = [p.get(SiteCfg.Attr.ElemTag) == SiteCfg.Types.Punct for p in preterminals]
            for i in false_indices(is_puncts):
                start, end = expand_to_neighboring_punct(i, is_puncts)
                if retokenize(i, start, end, terminals, preterminals,
                              preterminal_parents, passage.ID, tokenizer,
                              state,
                              cw, words_set):
                    ever_changed = changed = True
                    break
            if not changed:
                break
    return from_site(elem) if ever_changed else None


def read_dict(file):
    if file is None:
        return None
    with open(file, "r", encoding="utf-8") as file:
        words_to_change = dict()
        new_case = True
        cur_case = None
        for line in file:
            if new_case:
                cur_case = line.strip()
                new_case = False
                continue
            if line.strip() == "----":
                new_case = True
                continue
            words_to_change[line.strip()] = cur_case
    print(words_to_change)
    return words_to_change


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    words_set = read_dict(args.words_set)
    with open(args.logfile, "w", newline="", encoding="utf-8") as outfile:
        cw = csv.writer(outfile)
        for passage in get_passages_with_progress_bar(args.filenames, "Fixing tokenization"):
            fixed = fix_tokenization(passage, words_set, lang=args.lang, cw=cw)
            if fixed is not None:
                outfile.flush()
                normalize(fixed)
                write_passage(fixed, outdir=args.outdir, binary=args.binary, prefix=args.prefix, verbose=args.verbose)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to convert")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-O", "--logfile", default="fix_tokenization.csv", help="output log file")
    argparser.add_argument("-l", "--lang", default="en", help="language two-letter code for sentence model")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    argparser.add_argument("-s", "--words-set", default=None, help="filename to read the set of words from. each "
                                                                   "section starts with headline of the fix required "
                                                                   "(watch set format), followed by the words to "
                                                                   "fix. sections are separated by ---- line.")
    main(argparser.parse_args())
