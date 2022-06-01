from itertools import repeat

import os
import pytest
from functools import partial
from io import StringIO

from ucca import core, layer0, layer1, convert
from ucca.evaluation import evaluate, LABELED, UNLABELED, WEAK_LABELED
from ucca.validation import validate
from .conftest import PASSAGES, load_xml

PRIMARY = "primary"
REMOTE = "remote"

"""Tests the evaluation module functions and classes."""


def passage1():
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 20 terminals (1-20), #10 and #20 are punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0)) for i in range(1, 21)]

    # Linker #1 with terminal 1
    link1 = l1.add_fnode(None, layer1.EdgeTags.Linker)
    link1.add(layer1.EdgeTags.Terminal, terms[0])

    # Scene #1: [[2 3 4 5 P] [6 7 8 9 A] [10 U] H]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
    a1 = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
    p1.add(layer1.EdgeTags.Terminal, terms[1])
    p1.add(layer1.EdgeTags.Terminal, terms[2])
    p1.add(layer1.EdgeTags.Terminal, terms[3])
    p1.add(layer1.EdgeTags.Terminal, terms[4])
    a1.add(layer1.EdgeTags.Terminal, terms[5])
    a1.add(layer1.EdgeTags.Terminal, terms[6])
    a1.add(layer1.EdgeTags.Terminal, terms[7])
    a1.add(layer1.EdgeTags.Terminal, terms[8])
    l1.add_punct(ps1, terms[9])

    # Scene #23: [[11 12 13 14 15 H] [16 L] [17 18 19 H] H]
    # Scene #2: [[11 12 13 14 A] [15 D]]
    ps23 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    ps2 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)
    a2 = l1.add_fnode(ps2, layer1.EdgeTags.Participant)
    a2.add(layer1.EdgeTags.Terminal, terms[10])
    a2.add(layer1.EdgeTags.Terminal, terms[11])
    a2.add(layer1.EdgeTags.Terminal, terms[12])
    a2.add(layer1.EdgeTags.Terminal, terms[13])
    d2 = l1.add_fnode(ps2, layer1.EdgeTags.Adverbial)
    d2.add(layer1.EdgeTags.Terminal, terms[14])

    # Linker #2: [16 L]
    link2 = l1.add_fnode(ps23, layer1.EdgeTags.Linker)
    link2.add(layer1.EdgeTags.Terminal, terms[15])

    # Scene #3: [[16 17 S] [18 A] (implicit participant) H]
    ps3 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)
    p3 = l1.add_fnode(ps3, layer1.EdgeTags.State)
    p3.add(layer1.EdgeTags.Terminal, terms[16])
    p3.add(layer1.EdgeTags.Terminal, terms[17])
    a3 = l1.add_fnode(ps3, layer1.EdgeTags.Participant)
    a3.add(layer1.EdgeTags.Terminal, terms[18])
    l1.add_fnode(ps3, layer1.EdgeTags.Participant, implicit=True)

    # Punctuation #20 - not under a scene
    l1.add_punct(None, terms[19])

    # adding remote argument to scene #1, remote process to scene #2
    # creating linkages L1->H1, H2<-L2->H3
    l1.add_remote(ps1, layer1.EdgeTags.Participant, d2)
    l1.add_remote(ps2, layer1.EdgeTags.Process, p1)
    l1.add_linkage(link1, ps1)
    l1.add_linkage(link2, ps2, ps3)

    return p


def passage2():
    p = core.Passage("2")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 20 terminals (1-20), #10 and #20 are punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0)) for i in range(1, 21)]

    # Linker #1 with terminal 1
    link1 = l1.add_fnode(None, layer1.EdgeTags.Linker)  # true
    link1.add(layer1.EdgeTags.Terminal, terms[0])

    # Scene #1: [[2 3 4 5 P] [6 7 8 9 A] [10 U] H]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)  # true
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)  # true
    a1 = l1.add_fnode(ps1, layer1.EdgeTags.Participant)  # true
    p1.add(layer1.EdgeTags.Terminal, terms[1])
    p1.add(layer1.EdgeTags.Terminal, terms[2])
    p1.add(layer1.EdgeTags.Terminal, terms[3])
    p1.add(layer1.EdgeTags.Terminal, terms[4])
    a1.add(layer1.EdgeTags.Terminal, terms[5])
    a1.add(layer1.EdgeTags.Terminal, terms[6])
    a1.add(layer1.EdgeTags.Terminal, terms[7])
    a1.add(layer1.EdgeTags.Terminal, terms[8])
    l1.add_punct(ps1, terms[9])

    # Scene #23: [[11 12 13 14 15 H] [16 L] [17 18 19 H] H]
    # Scene #2: [[11 12 13 14 H] [15 E]]
    ps23 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)  # true
    ps2 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)  # true
    a2 = l1.add_fnode(ps2, layer1.EdgeTags.ParallelScene)  # false
    a2.add(layer1.EdgeTags.Terminal, terms[10])
    a2.add(layer1.EdgeTags.Terminal, terms[11])
    a2.add(layer1.EdgeTags.Terminal, terms[12])
    a2.add(layer1.EdgeTags.Terminal, terms[13])
    d2 = l1.add_fnode(ps1, layer1.EdgeTags.Elaborator)  # false
    d2.add(layer1.EdgeTags.Terminal, terms[14])

    # Linker #2: [16 L]
    link2 = l1.add_fnode(ps23, layer1.EdgeTags.Linker)  # true
    link2.add(layer1.EdgeTags.Terminal, terms[15])

    # Scene #3: [[16 17 P] [18 A] (implicit participant) H]
    ps3 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)  # true
    p3 = l1.add_fnode(ps3, layer1.EdgeTags.Process)  # false
    p3.add(layer1.EdgeTags.Terminal, terms[16])
    p3.add(layer1.EdgeTags.Terminal, terms[17])
    a3 = l1.add_fnode(ps3, layer1.EdgeTags.Participant)  # true
    a3.add(layer1.EdgeTags.Terminal, terms[18])
    l1.add_fnode(ps3, layer1.EdgeTags.Participant, implicit=True)

    # Punctuation #20 - not under a scene
    l1.add_punct(None, terms[19])

    # adding remote argument to scene #1, remote process to scene #2
    # creating linkages L1->H1, H2<-L2->H3
    l1.add_remote(ps1, layer1.EdgeTags.Participant, d2)
    l1.add_remote(ps1, layer1.EdgeTags.Participant, a3)
    l1.add_remote(ps2, layer1.EdgeTags.State, p1)
    l1.add_linkage(link1, ps1)
    l1.add_linkage(link2, ps2, ps3)

    return p


def simple1():
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 5 terminals (1-5), #5 is punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i == 5)) for i in range(1, 6)]

    # Scene #1: [H [P 1] [A 2]]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
    a = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
    p1.add(layer1.EdgeTags.Terminal, terms[0])
    a.add(layer1.EdgeTags.Terminal, terms[1])

    # Linker #1 with terminal 3
    link1 = l1.add_fnode(None, layer1.EdgeTags.Linker)
    link1.add(layer1.EdgeTags.Terminal, terms[2])

    # Scene #2: [H [A* 2] [S 4]]
    ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p2 = l1.add_fnode(ps2, layer1.EdgeTags.State)
    p2.add(layer1.EdgeTags.Terminal, terms[3])
    l1.add_fnode(ps2, layer1.EdgeTags.Participant, implicit=True)  # implicit should not affect evaluation

    # Punctuation #5 - not under a scene
    l1.add_punct(ps2, terms[4])  # punctuation should not affect evaluation

    # adding remote argument to scene #2
    l1.add_remote(ps2, layer1.EdgeTags.Participant, a)
    l1.add_linkage(link1, ps1, ps2)  # linkage should not affect evaluation

    return p


def simple2():
    p = core.Passage("2")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 5 terminals (1-5), #5 is punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i == 5)) for i in range(1, 6)]

    # Scene #1: [H [S 1] [D 2]]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.State)
    a = l1.add_fnode(ps1, layer1.EdgeTags.Adverbial)
    p1.add(layer1.EdgeTags.Terminal, terms[0])
    a.add(layer1.EdgeTags.Terminal, terms[1])

    # Linker #1 with terminal 3
    link1 = l1.add_fnode(None, layer1.EdgeTags.Linker)
    link1.add(layer1.EdgeTags.Terminal, terms[2])

    # Scene #2: [H [A* 2] [S 4]]
    ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p2 = l1.add_fnode(ps2, layer1.EdgeTags.State)
    p2.add(layer1.EdgeTags.Terminal, terms[3])

    # Punctuation #5 - not under a scene
    l1.add_punct(None, terms[4])

    # adding remote argument to scene #2
    l1.add_remote(ps2, layer1.EdgeTags.Adverbial, a)

    return p


def function1():
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 5 terminals (1-5), #5 is punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i == 5)) for i in range(1, 6)]

    # Scene #1: [H [P 1] [A 2]]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
    a = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
    p1.add(layer1.EdgeTags.Terminal, terms[0])
    a.add(layer1.EdgeTags.Terminal, terms[1])

    # Function #1 with terminal 3 - its location should not affect evaluation
    f = l1.add_fnode(None, layer1.EdgeTags.Function)
    f.add(layer1.EdgeTags.Terminal, terms[2])

    # Scene #2: [H [A* 2] [S 4]]
    ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p2 = l1.add_fnode(ps2, layer1.EdgeTags.State)
    p2.add(layer1.EdgeTags.Terminal, terms[3])
    l1.add_fnode(ps2, layer1.EdgeTags.Participant, implicit=True)  # implicit should not affect evaluation

    # Punctuation #5 - not under a scene
    l1.add_punct(ps2, terms[4])  # punctuation should not affect evaluation

    # adding remote argument to scene #2
    l1.add_remote(ps2, layer1.EdgeTags.Participant, a)

    return p


def function2():
    p = core.Passage("2")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 5 terminals (1-5), #5 is punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i == 5)) for i in range(1, 6)]

    # Scene #1: [H [S 1] [D 2] [F 2]]
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.State)
    a = l1.add_fnode(ps1, layer1.EdgeTags.Adverbial)
    p1.add(layer1.EdgeTags.Terminal, terms[0])
    a.add(layer1.EdgeTags.Terminal, terms[1])
    f = l1.add_fnode(ps1, layer1.EdgeTags.Function)
    f.add(layer1.EdgeTags.Terminal, terms[2])

    # Scene #2: [H [A* 2] [S 4]]
    ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p2 = l1.add_fnode(ps2, layer1.EdgeTags.State)
    p2.add(layer1.EdgeTags.Terminal, terms[3])

    # Punctuation #5 - not under a scene
    l1.add_punct(None, terms[4])

    # adding remote argument to scene #2
    l1.add_remote(ps2, layer1.EdgeTags.Adverbial, a)

    return p


def check_primary_remote(scores, expected):
    for (labeled, construction), score in sorted(expected.items()) if hasattr(expected, "items") else \
            zip([(l, c) for l, e in sorted(scores.evaluators.items()) for c in e.results], repeat(expected)):
        buf = StringIO()
        scores.print(file=buf)
        assert score == pytest.approx(scores[labeled][construction].f1), "%s_%s_f1\n%s" % (construction, labeled,
                                                                                           buf.getvalue())


@pytest.mark.parametrize("create", PASSAGES + (passage1, passage2, simple1, simple2, function1, function2))
@pytest.mark.parametrize("units", (True, False), ids=("units", ""))
@pytest.mark.parametrize("errors", (True, False), ids=("errors", ""))
@pytest.mark.parametrize("normalize", (True, False), ids=("normalize", ""))
def test_evaluate_self(create, units, errors, normalize):
    p = create()
    scores = evaluate(p, p, units=units, errors=errors, normalize=normalize)
    assert 1.0 == scores.average_f1()
    for eval_type, results in sorted(scores.evaluators.items()):
        for construction, stats in results.results.items():
            assert 1.0 == stats.f1, (eval_type, construction)
            assert 1.0 == stats.p, (eval_type, construction)
            assert 1.0 == stats.r, (eval_type, construction)
    check_primary_remote(scores, 1.0)


@pytest.mark.parametrize("create1, create2, f1", (
                                 (passage1, passage2, {(LABELED, PRIMARY): 0.5, (LABELED, REMOTE): 0.4,
                                                       (UNLABELED, PRIMARY): 0.75, (UNLABELED, REMOTE): 0.8,
                                                       (WEAK_LABELED, PRIMARY): 7/12, (WEAK_LABELED, REMOTE): 0.8}),
                                 (simple1, simple2, {(LABELED, PRIMARY): 0.6, (LABELED, REMOTE): 0,
                                                     (UNLABELED, PRIMARY): 1, (UNLABELED, REMOTE): 1,
                                                     (WEAK_LABELED, PRIMARY): 0.8, (WEAK_LABELED, REMOTE): 0}),
                                 (function1, function2, {(LABELED, PRIMARY): 0.6, (LABELED, REMOTE): 0,
                                                         (UNLABELED, PRIMARY): 1, (UNLABELED, REMOTE): 1,
                                                         (WEAK_LABELED, PRIMARY): 0.8, (WEAK_LABELED, REMOTE): 0}),
                                 tuple(partial(convert.from_standard, load_xml(
                                     os.path.join("test_files", "%s.xml" % f))) for f in ("120_parsed", "standard3")) +
                                 ({(LABELED, PRIMARY): 3/14, (LABELED, REMOTE): 0,
                                   (UNLABELED, PRIMARY): 0.5, (UNLABELED, REMOTE): 0,
                                   (WEAK_LABELED, PRIMARY): 3/14, (WEAK_LABELED, REMOTE): 0},)
                         ))
@pytest.mark.parametrize("units", (True, False), ids=("units", ""))
@pytest.mark.parametrize("errors", (True, False), ids=("errors", ""))
def test_evaluate(create1, create2, f1, units, errors):
    p1 = create1()
    p2 = create2()
    validation_errors_before = [list(validate(p, linkage=False)) for p in (p1, p2)]
    scores = evaluate(p1, p2, units=units, errors=errors)
    validation_errors_after = [list(validate(p, linkage=False)) for p in (p1, p2)]
    for before, after in zip(validation_errors_before, validation_errors_after):
        if not before:
            assert not after
    check_primary_remote(scores, f1)
