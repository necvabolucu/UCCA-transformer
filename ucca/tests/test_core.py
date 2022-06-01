"""Testing code for the ucca package, unit-testing only."""

import pytest

from ucca import core, layer0, layer1
from .conftest import basic, PASSAGES


def test_creation():
    p = basic()

    assert p.ID == "1"
    assert p.root == p
    assert p.attrib.copy() == {}
    assert p.layer("1").ID == "1"
    assert p.layer("2").ID == "2"
    with pytest.raises(KeyError):
        p.layer("3")

    l1 = p.layer("1")
    l2 = p.layer("2")
    assert l1.root == p
    assert l2.attrib["test"]
    assert l1.orderkey != l2.orderkey
    assert [x.ID for x in l1.all] == ["1.1", "1.2", "1.3"]
    assert [x.ID for x in l1.heads] == ["1.2"]
    assert [x.ID for x in l2.all] == ["2.2", "2.1"]
    assert [x.ID for x in l2.heads] == ["2.2", "2.1"]

    node11, node12, node13 = l1.all
    node22, node21 = l2.all
    assert node11.ID == "1.1"
    assert node11.root == p
    assert node11.layer.ID == "1"
    assert node11.tag == "1"
    assert len(node11) == 0
    assert node11.parents == [node12, node21, node22]
    assert node13.parents == [node12, node22]
    assert node13.attrib.copy() == {"node": True}
    assert len(node12) == 2
    assert node12.children == [node13, node11]
    assert node12[0].attrib.copy() == {"edge": True}
    assert node12.parents == [node22, node21]
    assert node21[0].ID == "2.1->1.1"
    assert node21[1].ID == "2.1->1.2"
    assert node22[0].ID == "2.2->1.1"
    assert node22[1].ID == "2.2->1.2"
    assert node22[2].ID == "2.2->1.3"


def test_modifying():
    p = basic()
    l1, l2 = p.layer("1"), p.layer("2")
    node11, node12, node13 = l1.all
    node22, node21 = l2.all

    # Testing attribute changes
    p.attrib["passage"] = 1
    assert p.attrib.copy() == {"passage": 1}
    del l2.attrib["test"]
    assert l2.attrib.copy() == {}
    node13.attrib[1] = 1
    assert node13.attrib.copy() == {"node": True, 1: 1}
    assert len(node13.attrib) == 2
    assert node13.attrib.get("node")
    assert node13.attrib.get("missing") is None

    # Testing Node changes
    node14 = core.Node(ID="1.4", root=p, tag="4")
    node15 = core.Node(ID="1.5", root=p, tag="5")
    assert l1.all == [node11, node12, node13, node14, node15]
    assert l1.heads == [node12, node14, node15]
    node15.add("test", node11)
    assert node11.parents == [node12, node15, node21, node22]
    node21.remove(node12)
    node21.remove(node21[0])
    assert len(node21) == 0
    assert node12.parents == [node22]
    assert node11.parents == [node12, node15, node22]
    node14.add("test", node15)
    assert l1.heads == [node12, node14]
    node12.destroy()
    assert l1.heads == [node13, node14]
    assert node22.children == [node11, node13]

    node22.tag = "x"
    node22[0].tag = "testx"
    assert node22.tag == "x"
    assert node22[0].tag == "testx"


def test_equals():
    p1 = core.Passage("1")
    p2 = core.Passage("2")
    p1l0 = layer0.Layer0(p1)
    p2l0 = layer0.Layer0(p2)
    p1l1 = layer1.Layer1(p1)
    p2l1 = layer1.Layer1(p2)
    assert (p1.equals(p2) and p2.equals(p1))

    # Checks basic passage equality and Attrib/tag/len differences
    p1l0.add_terminal("0", False)
    p1l0.add_terminal("1", False)
    p1l0.add_terminal("2", False)
    p2l0.add_terminal("0", False)
    p2l0.add_terminal("1", False)
    p2l0.add_terminal("2", False)
    assert (p1.equals(p2) and p2.equals(p1))
    pnct2 = p2l0.add_terminal("3", True)
    assert not (p1.equals(p2) or p2.equals(p1))
    temp = p1l0.add_terminal("3", False)
    assert not (p1.equals(p2) or p2.equals(p1))
    temp.destroy()
    pnct1 = p1l0.add_terminal("3", True)
    assert (p1.equals(p2) and p2.equals(p1))

    # Check Edge and node equality
    ps1 = p1l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    assert not (p1.equals(p2) or p2.equals(p1))
    ps2 = p2l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    assert (p1.equals(p2) and p2.equals(p1))
    p1l1.add_fnode(ps1, layer1.EdgeTags.Participant)
    assert not (p1.equals(p2) or p2.equals(p1))
    assert (ps1.equals(ps2, recursive=False))
    p2l1.add_fnode(ps2, layer1.EdgeTags.Process)
    assert not (p1.equals(p2) or p2.equals(p1))
    p2l1.add_fnode(ps2, layer1.EdgeTags.Participant)
    assert not (p1.equals(p2) or p2.equals(p1))
    p1l1.add_fnode(ps1, layer1.EdgeTags.Process)
    assert (p1.equals(p2) and p2.equals(p1))
    assert not (p1.equals(p2, ordered=True) or
                p2.equals(p1, ordered=True))
    p1l1.add_fnode(ps1, layer1.EdgeTags.Adverbial, implicit=True)
    ps2d3 = p2l1.add_fnode(ps2, layer1.EdgeTags.Adverbial)
    assert not (p1.equals(p2) or p2.equals(p1))
    ps2d3.attrib["implicit"] = True
    assert (p1.equals(p2) and p2.equals(p1))
    ps2[2].attrib["remote"] = True
    assert not (p1.equals(p2) or p2.equals(p1))
    ps1[2].attrib["remote"] = True
    assert (p1.equals(p2) and p2.equals(p1))
    p1l1.add_punct(None, pnct1)
    assert not (p1.equals(p2) or p2.equals(p1))
    p2l1.add_punct(None, pnct2)
    assert (p1.equals(p2) and p2.equals(p1))
    core.Layer("2", p1)
    assert not (p1.equals(p2) or p2.equals(p1))


@pytest.mark.parametrize("create", PASSAGES)
def test_copying(create):
    # we don't need such a complex passage, but it will work anyway
    p1 = create()

    p2 = p1.copy(())
    assert p1.ID == p2.ID
    assert (p1.attrib.equals(p2.attrib))
    assert p1.extra == p2.extra
    assert p1.frozen == p2.frozen

    l0id = layer0.LAYER_ID
    p2 = p1.copy([l0id])
    assert (p1.layer(l0id).equals(p2.layer(l0id)))


def test_iteration():
    p = basic()
    l1, l2 = p.layer("1"), p.layer("2")
    node11, node12, node13 = l1.all
    node22, node21 = l2.all

    assert list(node11.iter()) == [node11]
    assert not list(node11.iter(obj="edges"))
    assert not list(node13.iter(key=lambda x: x.tag != "3"))
    assert list(node12.iter()) == [node12, node13, node11]
    assert list(x.ID for x in node12.iter(obj="edges")) == ["1.2->1.3", "1.2->1.1"]
    assert list(node21.iter(duplicates=True)) == [node21, node11, node12, node13, node11]
    assert list(node21.iter()) == [node21, node11, node12, node13]
    assert list(node22.iter(method="bfs", duplicates=True)) == [node22, node11, node12, node13, node13, node11]
