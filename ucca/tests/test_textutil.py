import pytest

from ucca import layer0, convert, textutil
from .conftest import crossing, multi_sent, l1_passage, discontiguous, empty, PASSAGES

"""Tests the textutil module functions and classes."""


@pytest.mark.parametrize("create, breaks", (
        (multi_sent, [4, 7, 11]),
        (crossing, [3, 7]),
        (discontiguous, [20]),
        (l1_passage, [20]),
        (empty, []),
))
def test_break2sentences(create, breaks):
    """Tests identifying correctly sentence ends. """
    assert textutil.break2sentences(create()) == breaks


def test_word_vectors():
    vectors, dim = textutil.get_word_vectors()
    for word, vector in vectors.items():
        assert len(vector) == dim, "Vector dimension for %s is %d != %d" % (word, len(vector), dim)


@pytest.mark.parametrize("create", PASSAGES)
@pytest.mark.parametrize("as_array", (True, False), ids=("array", "extra"))
def test_annotate_passage(create, as_array):
    passage = create()
    textutil.annotate(passage, as_array=as_array)
    for p in passage, convert.from_standard(convert.to_standard(passage)):
        assert textutil.is_annotated(p, as_array=as_array), "Passage %s is not annotated" % passage.ID
        for terminal in p.layer(layer0.LAYER_ID).all:
            if as_array:
                assert terminal.tok is not None, "Terminal %s has no annotation" % terminal
                assert len(terminal.tok) == len(textutil.Attr)
            else:
                for attr in textutil.Attr:
                    assert attr.key in terminal.extra, "Terminal %s has no %s" % (terminal, attr.name)


@pytest.mark.parametrize("as_array", (True, False), ids=("array", "extra"))
@pytest.mark.parametrize("convert_and_back", (True, False), ids=("convert", "direct"))
def test_annotate_all(as_array, convert_and_back):
    passages = [create() for create in PASSAGES]
    list(textutil.annotate_all(passages))
    for passage, compare in textutil.annotate_all(((p, p) for p in passages), as_array=as_array, as_tuples=True):
        assert passage is compare
        p = (passage, convert.from_standard(convert.to_standard(passage)))[convert_and_back]
        assert textutil.is_annotated(p, as_array=as_array), "Passage %s is not annotated" % passage.ID
        for terminal in p.layer(layer0.LAYER_ID).all:
            if as_array:
                assert terminal.tok is not None, "Terminal %s in passage %s has no annotation" % (terminal, passage.ID)
                assert len(terminal.tok) == len(textutil.Attr)
            else:
                for attr in textutil.Attr:
                    assert attr.key in terminal.extra, "Terminal %s in passage %s has no %s" % (
                        terminal, passage.ID, attr.name)


def assert_spacy_not_loaded(*args, **kwargs):
    del args, kwargs
    assert False, "Should not load spaCy when passage is pre-annotated"


@pytest.mark.parametrize("create", PASSAGES)
@pytest.mark.parametrize("as_array", (True, False), ids=("array", "extra"))
@pytest.mark.parametrize("convert_and_back", (True, False), ids=("convert", "direct"))
@pytest.mark.parametrize("partial", (True, False), ids=("partial", "full"))
def test_preannotate_passage(create, as_array, convert_and_back, partial, monkeypatch):
    if not partial:
        monkeypatch.setattr(textutil, "get_nlp", assert_spacy_not_loaded)
    passage = create()
    l0 = passage.layer(layer0.LAYER_ID)
    attr_values = list(range(10, 10 + len(textutil.Attr)))
    if partial:
        attr_values[textutil.Attr.ENT_TYPE.value] = ""
    if as_array:
        l0.extra["doc"] = [len(p) * [attr_values] for p in textutil.break2paragraphs(passage, return_terminals=True)]
    else:
        for terminal in l0.all:
            for attr, value in zip(textutil.Attr, attr_values):
                if value:
                    terminal.extra[attr.key] = value
    passage = (passage, convert.from_standard(convert.to_standard(passage)))[convert_and_back]
    if not partial:
        assert textutil.is_annotated(passage, as_array=as_array), "Passage %s is not pre-annotated" % passage.ID
    textutil.annotate(passage, as_array=as_array)
    assert textutil.is_annotated(passage, as_array=as_array), "Passage %s is not annotated" % passage.ID
    for terminal in l0.all:
        for i, (attr, value) in enumerate(zip(textutil.Attr, attr_values)):
            if value:
                assert (terminal.tok[i] if as_array else terminal.extra.get(attr.key)) == value, \
                    "Terminal %s has wrong %s" % (terminal, attr.name)
