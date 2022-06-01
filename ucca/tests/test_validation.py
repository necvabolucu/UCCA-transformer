import pytest

from ucca import layer1
from ucca.validation import validate
from .conftest import loaded, loaded_valid, multi_sent, crossing, discontiguous, l1_passage, empty, \
    create_passage, attach_terminals

"""Tests the validation module functions and classes."""


def unary_punct():
    p, l1, terms = create_passage(3, 3)
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
    a1 = l1.add_fnode(h1, layer1.EdgeTags.Participant)
    l1.add_punct(h1, terms[2])
    attach_terminals(terms, p1, a1)
    return p


def binary_punct():
    p, l1, terms = create_passage(4, 3)
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
    a1 = l1.add_fnode(h1, layer1.EdgeTags.Participant)
    l1.add_punct(h1, terms[2]).add(layer1.EdgeTags.Terminal, terms[3])
    attach_terminals(terms, p1, a1)
    return p


def unary_punct_under_fn():
    p, l1, terms = create_passage(3, 3)
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
    a1 = l1.add_fnode(h1, layer1.EdgeTags.Participant)
    attach_terminals(terms, p1, a1, h1)
    return p


def punct_under_unanalyzable_fn():
    p, l1, terms = create_passage(3, 2)
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
    attach_terminals(terms, p1, p1, p1)
    return p


@pytest.mark.parametrize("create, valid", (
        (loaded, False),
        (loaded_valid, True),
        (multi_sent, True),
        (crossing, True),
        (discontiguous, True),
        (l1_passage, True),
        (empty, False),
        (unary_punct, True),
        (binary_punct, True),
        (unary_punct_under_fn, False),
        (punct_under_unanalyzable_fn, True),
))
def test_evaluate_self(create, valid):
    p = create()
    errors = list(validate(p))
    if valid:
        assert not errors, p
    else:
        assert errors, p
