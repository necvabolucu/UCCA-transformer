from collections import OrderedDict

import pytest

from ucca import textutil
from ucca.constructions import CATEGORIES_NAME, DEFAULT, CONSTRUCTIONS, extract_candidates
from .conftest import PASSAGES, loaded, loaded_valid, multi_sent, crossing, discontiguous, l1_passage, empty

"""Tests the constructions module functions and classes."""


def assert_spacy_not_loaded(*args, **kwargs):
    del args, kwargs
    assert False, "Should not load spaCy when passage is pre-annotated"


def extract_and_check(p, constructions=None, expected=None):
    d = OrderedDict((construction, [candidate.edge for candidate in candidates]) for construction, candidates in
                    extract_candidates(p, constructions=constructions).items() if candidates)
    if expected is not None:
        hist = {c.name: len(e) for c, e in d.items()}
        assert hist == expected, " != ".join(",".join(sorted(h)) for h in (hist, expected))


@pytest.mark.parametrize("create, expected", (
        (loaded, {'P': 1, 'remote': 1, 'E': 3, 'primary': 15, 'U': 2, 'F': 1, 'C': 3, 'A': 1, 'D': 1, 'L': 2, 'mwe': 2,
                  'H': 5}),
        (loaded_valid, {'P': 1, 'remote': 1, 'E': 3, 'primary': 15, 'U': 2, 'F': 1, 'C': 3, 'A': 1, 'D': 1, 'L': 2,
                        'mwe': 2, 'H': 5}),
        (multi_sent, {'U': 4, 'P': 3, 'mwe': 2, 'H': 3, 'primary': 6}),
        (crossing, {'U': 3, 'P': 2, 'remote': 1, 'mwe': 1, 'H': 2, 'primary': 3}),
        (discontiguous, {'G': 2, 'U': 2, 'remote': 1, 'E': 2, 'primary': 13, 'P': 3, 'F': 1, 'C': 1, 'A': 3, 'D': 2,
                         'mwe': 6, 'H': 3}),
        (l1_passage, {'P': 2, 'mwe': 4, 'H': 4, 'primary': 12, 'U': 2, 'A': 5, 'D': 1, 'L': 2, 'remote': 2, 'S': 1}),
        (empty, {}),
))
def test_extract_all(create, expected):
    extract_and_check(create(), constructions=CONSTRUCTIONS, expected=expected)


@pytest.mark.parametrize("create", PASSAGES)
@pytest.mark.parametrize("constructions", (DEFAULT, [CATEGORIES_NAME]), ids=("default", CATEGORIES_NAME))
def test_extract(create, constructions, monkeypatch):
    monkeypatch.setattr(textutil, "get_nlp", assert_spacy_not_loaded)
    extract_and_check(create(), constructions=constructions)
