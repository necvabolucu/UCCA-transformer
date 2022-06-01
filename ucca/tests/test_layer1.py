from ucca import layer1
from .conftest import l1_passage, discontiguous

"""Tests layer1 module functionality and correctness."""


def test_creation():
    p = l1_passage()
    head = p.layer("1").heads[0]
    assert [x.tag for x in head] == ["L", "H", "H", "U"]
    assert [x.child.position for x in head.children[0]] == [1]
    assert [x.tag for x in head.children[1]] == ["P", "A", "U", "A"]
    assert [x.child.position for x in head.children[1].children[0]] == [2, 3, 4, 5]
    assert [x.child.position for x in head.children[1].children[1]] == [6, 7, 8, 9]
    assert [x.child.position for x in head.children[1].children[2]] == [10]
    assert (head.children[1][3].attrib.get("remote"))


def test_fnodes():
    p = l1_passage()
    l0 = p.layer("0")
    l1 = p.layer("1")

    terms = l0.all
    head, lkg1, lkg2 = l1.heads
    link1, ps1, ps23, punct2 = head.children
    p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
    ps2, link2, ps3 = ps23.children
    a2, d2 = [x.child for x in ps2 if not x.attrib.get("remote")]
    p3, a3, a4 = ps3.children

    assert lkg1.relation == link1
    assert lkg1.arguments == [ps1]
    assert ps23.process is None
    assert ps2.process == p1
    assert ps1.participants == [a1, d2]
    assert ps3.participants == [a3, a4]

    assert ps1.get_terminals() == terms[1:10]
    assert ps1.get_terminals(punct=False, remotes=True) == terms[1:9] + terms[14:15]
    assert ps1.end_position == 10
    assert ps2.start_position == 11
    assert ps3.start_position == 17
    assert a4.start_position == -1
    assert ps23.to_text() == "11 12 13 14 15 16 17 18 19"

    assert ps1.fparent == head
    assert link2.fparent == ps23
    assert ps2.fparent == ps23
    assert d2.fparent == ps2


def test_layer1():
    p = l1_passage()
    l1 = p.layer("1")

    head, lkg1, lkg2 = l1.heads
    link1, ps1, ps23, punct2 = head.children
    p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
    ps2, link2, ps3 = ps23.children

    assert l1.top_scenes == [ps1, ps2, ps3]
    assert l1.top_linkages == [lkg1, lkg2]

    # adding scene #23 to linkage #1, which makes it non top-level as
    # scene #23 isn't top level
    lkg1.add(layer1.EdgeTags.LinkArgument, ps23)
    assert l1.top_linkages == [lkg2]

    # adding process to scene #23, which makes it top level and discards
    # "top-levelness" from scenes #2 + #3
    l1.add_remote(ps23, layer1.EdgeTags.Process, p1)
    assert l1.top_scenes == [ps1, ps23]
    assert l1.top_linkages == [lkg1, lkg2]

    # Changing the process tag of scene #1 to A and back, validate that
    # top scenes are updates accordingly
    p_edge = [e for e in ps1 if e.tag == layer1.EdgeTags.Process][0]
    p_edge.tag = layer1.EdgeTags.Participant
    assert l1.top_scenes == [ps23]
    assert l1.top_linkages == [lkg2]
    p_edge.tag = layer1.EdgeTags.Process
    assert l1.top_scenes == [ps1, ps23]
    assert l1.top_linkages == [lkg1, lkg2]


def test_str():
    p = l1_passage()
    assert [str(x) for x in p.layer("1").heads] == \
           ["[L 1] [H [P 2 3 4 5] [A 6 7 8 9] [U 10] "
            "... [A* 15] ] [H [H [P* 2 3 4 5] [A 11 12 "
            "13 14] [D 15] ] [L 16] [H [A IMPLICIT] [S "
            "17 18] [A 19] ] ] [U 20] ",
            "1.2-->1.3", "1.11-->1.8,1.12"]


def test_destroy():
    p = l1_passage()
    l1 = p.layer("1")

    head, lkg1, lkg2 = l1.heads
    link1, ps1, ps23, punct2 = head.children
    p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
    ps2, link2, ps3 = ps23.children

    ps1.destroy()
    assert head.children == [link1, ps23, punct2]
    assert p1.parents == [ps2]
    assert not a1.parents
    assert not punct1.parents


def test_discontiguous():
    """Tests FNode.discontiguous and FNode.get_sequences"""
    p = discontiguous()
    l1 = p.layer("1")
    head = l1.heads[0]
    ps1, ps2, ps3 = head.children
    d1, a1, p1, f1 = ps1.children
    e1, c1, e2, g1 = d1.children
    d2, g2, p2, a2 = ps2.children
    t14, p3, a3 = ps3.children

    # Checking discontiguous property
    assert not ps1.discontiguous
    assert not d1.discontiguous
    assert not e1.discontiguous
    assert not e2.discontiguous
    assert c1.discontiguous
    assert g1.discontiguous
    assert a1.discontiguous
    assert p1.discontiguous
    assert not f1.discontiguous
    assert ps2.discontiguous
    assert not p2.discontiguous
    assert not a2.discontiguous
    assert not ps3.discontiguous
    assert not a3.discontiguous

    # Checking get_sequences -- should return only non-remote, non-implicit
    # stretches of terminals
    assert ps1.get_sequences() == [(1, 10)]
    assert d1.get_sequences() == [(1, 4)]
    assert e1.get_sequences() == [(1, 1)]
    assert e2.get_sequences() == [(3, 3)]
    assert c1.get_sequences() == [(2, 2), (4, 4)]
    assert a1.get_sequences() == [(5, 5), (8, 8)]
    assert p1.get_sequences() == [(6, 7), (10, 10)]
    assert f1.get_sequences() == [(9, 9)]
    assert ps2.get_sequences() == [(11, 14), (18, 20)]
    assert p2.get_sequences() == [(11, 14)]
    assert a2.get_sequences() == [(18, 20)]
    assert not d2.get_sequences()
    assert not g2.get_sequences()
    assert ps3.get_sequences() == [(15, 17)]
    assert a3.get_sequences() == [(16, 17)]
    assert not p3.get_sequences()
