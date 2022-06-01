from collections import OrderedDict
from itertools import chain

from ucca import textutil, layer0, layer1
from ucca.layer1 import EdgeTags, NodeTags


class Construction:
    def __init__(self, name, description, criterion, default=False):
        """
        :param name: short name
        :param description: long description
        :param criterion: predicate function to apply to a Candidate, saying if it is an instance of this construction
        :param default: whether this construction is included in evaluation by default
        """
        self.name = name
        self.description = description
        self.criterion = criterion
        self.default = default

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == (other.name if isinstance(other, Construction) else other)

    def __call__(self, candidate):
        if self.criterion(candidate):
            yield self

    @property
    def is_punct(self):
        return self.name in (EdgeTags.Punctuation, layer0.NodeTags.Punct, "punct")


CATEGORIES_NAME = "categories"
CATEGORY_DESCRIPTIONS = {v: k for k, v in EdgeTags.__dict__.items() if not k.startswith("_")}


class Categories(Construction):
    def __init__(self):
        super().__init__(CATEGORIES_NAME, description=None, criterion=None)

    def __call__(self, candidate):
        try:
            tags = candidate.edge.tags
        except AttributeError:
            tags = [candidate]
        for tag in tags:
            yield create_category_construction(tag)


def create_category_construction(tag):
    return Construction(tag, CATEGORY_DESCRIPTIONS.get(tag, tag), criterion=None)


def positions(terminals):
    return frozenset(t.position for t in terminals)


class Candidate:
    def __init__(self, edge, reference=None, reference_yield_tags=None, verbose=False):
        self.edge = edge
        self.out_tags = {t for e in edge.child for t in e.tags}
        self.reference = reference
        self.reference_yield_tags = reference_yield_tags
        self.verbose = verbose
        self.terminals = self.edge.child.get_terminals()
        self._terminal_yield = positions(self.terminals)
        self._terminal_yield_no_punct = positions(self.edge.child.get_terminals(punct=False))
        if self.reference is not None:
            self.terminals = [self.reference.by_id(t.ID) for t in self.terminals]
        self.extra = {}
        self.is_unary_child = self.edge.parent.incoming and (
                self._terminal_yield_no_punct == positions(self.edge.parent.get_terminals(punct=False)))

    def _annotate(self, attr=None):
        passage = self.edge.parent.root
        if not passage.extra.get("annotated"):
            textutil.annotate(passage, as_array=True, verbose=self.verbose)
            passage.extra["annotated"] = True
        if attr:
            ret = self.extra.get(attr)
            if ret is None:
                ret = self.extra[attr] = {t.get_annotation(attr, as_array=True) for t in self.terminals}
            return ret

    @property
    def remote(self):
        return self.edge.attrib.get("remote", False)

    @property
    def implicit(self):
        return self.edge.child.attrib.get("implicit", False)

    @property
    def excluded(self):
        return bool(EXCLUDED_EDGE_TAGS.intersection(self.edge.tags)) or self.edge.child.tag in EXCLUDED_NODE_TAGS

    @property
    def pos(self):
        return self._annotate(attr=textutil.Attr.POS)

    @property
    def dep(self):
        return self._annotate(attr=textutil.Attr.DEP)

    @property
    def heads(self):
        attr = textutil.Attr.HEAD
        ret = self.extra.get(attr)
        if ret is None:
            self._annotate()
            para_pos = {t.para_pos for t in self.terminals}
            ret = self.extra[attr] = {t for t in self.terminals if int(t.tok[attr]) not in para_pos}
        return ret

    @property
    def tokens(self):
        attr = "tokens"
        ret = self.extra.get(attr)
        if ret is None:
            ret = self.extra[attr] = {t.text.lower() for t in self.terminals}
        return ret

    def is_punct(self):
        return EdgeTags.Punctuation in self.edge.tags or self.edge.child.tag == NodeTags.Punctuation

    def is_primary(self):
        return not self.remote and not self.implicit and not self.is_punct()

    def is_remote(self):
        return self.remote and not self.implicit and not self.is_punct()

    def is_predicate(self):
        return bool({EdgeTags.Process, EdgeTags.State}.intersection(self.edge.tags)) and \
               self.out_tags <= {EdgeTags.Center, EdgeTags.Function, EdgeTags.Terminal} and \
               "to" not in self.tokens

    def constructions(self, constructions=None):
        for construction in constructions or [ALL_EDGES]:
            if construction.name == CATEGORIES_NAME and self.reference_yield_tags is not None:
                if not self.is_remote():
                    for terminal_yield, is_punct in (self._terminal_yield, True), \
                                                    (self._terminal_yield_no_punct, False):
                        for tag in self.reference_yield_tags.get(terminal_yield, ()):
                            for category_construction in construction(tag):
                                if category_construction.is_punct == is_punct:
                                    yield category_construction
            else:
                yield from construction(self)

    def terminal_yield(self, construction):
        return self._terminal_yield if construction.is_punct else self._terminal_yield_no_punct

    def __str__(self):
        return "[%s %s]" % (" ".join(self.edge.tags), self.edge.child)


EXCLUDED_EDGE_TAGS = {EdgeTags.LinkArgument, EdgeTags.LinkRelation, EdgeTags.Terminal}
EXCLUDED_NODE_TAGS = {NodeTags.Linkage, layer0.NodeTags.Word, layer0.NodeTags.Punct}

CONSTRUCTIONS = (
    Construction("primary", "Regular edges", Candidate.is_primary, default=True),
    Construction("remote", "Remote edges", Candidate.is_remote, default=True),
    Construction("aspectual_verbs", "Aspectual verbs",
                 lambda c: c.pos == {"VERB"} and EdgeTags.Adverbial in c.edge.tags),
    Construction("light_verbs", "Light verbs",
                 lambda c: c.pos == {"VERB"} and EdgeTags.Function in c.edge.tags),
    Construction("mwe", "Multi-word expressions",
                 lambda c: c.is_primary() and c.edge.child.tag == NodeTags.Foundational and len(
                     c.edge.child.terminals) > 1),  # Unanalyzable unit
    Construction("pred_nouns", "Predicate nouns",
                 lambda c: "ADJ" not in c.pos and "NOUN" in c.pos and c.is_predicate()),
    Construction("pred_adjs", "Predicate adjectives",
                 lambda c: "ADJ" in c.pos and "NOUN" not in c.pos and c.is_predicate()),
    Construction("expletives", "Expletives",
                 lambda c: c.tokens <= {"it", "there"} and EdgeTags.Function in c.edge.tags),
    Categories(),
)
PRIMARY = CONSTRUCTIONS[0]
CONSTRUCTION_BY_NAME = OrderedDict([(c.name, c) for c in CONSTRUCTIONS])
DEFAULT = OrderedDict((str(c), c) for c in CONSTRUCTIONS if c.default)
ALL_EDGES = Construction("all", "All edges", bool)


def add_argument(argparser, default=True):
    d = list(DEFAULT) if default else [n for n in CONSTRUCTION_BY_NAME if n not in DEFAULT]
    argparser.add_argument("--constructions", nargs="*", choices=CONSTRUCTION_BY_NAME, default=d, metavar="x",
                           help="construction types to include, out of {%s}" % ",".join(CONSTRUCTION_BY_NAME))


def get_by_name(name):
    return name if isinstance(name, Construction) else CATEGORY_DESCRIPTIONS.get(name) or CONSTRUCTION_BY_NAME[name]


def get_by_names(names=None):
    return list(map(get_by_name, names or ()))


def terminal_ids(passage):
    return {t.ID for t in passage.layer(layer0.LAYER_ID).all}


def diff_terminals(*passages):
    texts = [[t.text for t in p.layer(layer0.LAYER_ID).all] for p in passages]
    return [[t for t in texts[i] if t not in texts[j]] for i, j in ((0, 1), (1, 0))]


def verify_terminals_match(passage, reference):
    ids1, ids2 = terminal_ids(passage), terminal_ids(reference)
    assert ids1 == ids2, "Reference passage terminals do not match (%d != %d)\n" \
                         "Passage ID: %s\nReference ID: %s\nDifference:\n%s" % \
                         (len(terminal_ids(passage)), len(terminal_ids(reference)), passage.ID, reference.ID,
                          "\n".join(map(str, diff_terminals(passage, reference))))


def extract_candidates(passage, constructions=None, reference=None, reference_yield_tags=None, verbose=False):
    """
    Find candidate edges by constructions in UCCA passage.
    :param passage: Passage object to find constructions in
    :param constructions: list of constructions to include or None for all
    :param reference: Passage object to get POS tags from, and categories for fine-grained scores (default: `passage')
    :param reference_yield_tags: yield tags from reference passage for fine-grained evaluation:
                   dict: set of terminal indices (excluding punctuation) ->
                   list of edges of the Construction whose yield (excluding remotes and punctuation) is that set
    :param verbose: whether to print tagged text
    :return: dict of Construction -> list of corresponding Candidates
    """
    constructions = get_by_names(constructions)
    if reference is not None:
        verify_terminals_match(passage, reference)
    keys = []
    for construction in constructions:
        if construction.name == CATEGORIES_NAME:
            if reference_yield_tags:
                keys += list(map(create_category_construction, sorted(set(chain(*reference_yield_tags.values())))))
        else:
            keys.append(construction)
    extracted = OrderedDict((c, []) for c in keys)
    for node in passage.layer(layer1.LAYER_ID).all:
        for edge in node:
            candidate = Candidate(edge, reference or passage, reference_yield_tags, verbose=verbose)
            if not candidate.excluded:
                for construction in candidate.constructions(constructions):
                    extracted.setdefault(construction, []).append(candidate)
    return extracted


def create_passage_yields(p, *args, tags=True, **kwargs):
    """
    :param p: passage to find terminal yields of
    :param tags: instead of Candidates, map simply to their edge tags
    :returns: dict: Construction ->
                   dict: set of terminal indices (excluding punctuation) ->
                         list of Candidates whose yield (excluding remotes and punctuation) is that set
    """
    yield_candidates = OrderedDict()
    for construction, candidates in extract_candidates(p, *args, **kwargs).items():
        construction_yield_candidates = yield_candidates[construction] = {}
        for candidate in candidates:
            terminal_yield = candidate.terminal_yield(construction)
            # if terminal_yield:
            construction_yield_candidates.setdefault(terminal_yield, []).extend(
                candidate.edge.tags if tags else [candidate])
    return yield_candidates
