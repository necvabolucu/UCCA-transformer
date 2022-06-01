import operator
import xml.etree.ElementTree as ETree

from ucca import core, layer0, layer1, convert

"""Utilities for tests."""


def empty():
    p = core.Passage(ID="1")
    layer0.Layer0(p)
    layer1.Layer1(p)
    return p


def basic():
    """Creates a basic :class:`Passage` to tinker with.

    Passage structure is as follows:
        Layer1: order by ID, heads = [1.2], all = [1.1, 1.2, 1.3]
        Layer2: order by node unique ID descending,
                heads = all = [2.2, 2.1], attrib={"test": True}
        Nodes (tag):
            1.1 (1)
            1.3 (3), attrib={"node": True}
            1.2 (x), order by edge tag
                children: 1.3 Edge: tag=test1, attrib={"Edge": True}
                          1.1 Edge: tag=test2
            2.1 (2), children [1.1, 1.2] with edge tags [test, test2]
            2.2 (2), children [1.1, 1.2, 1.3] with tags [test, test1, test]

    """
    p = core.Passage(ID="1")
    core.Layer(ID="1", root=p)
    core.Layer(ID="2", root=p, attrib={"test": True},
               orderkey=lambda x: -1 * int(x.ID.split(".")[1]))

    # Order is explicitly different in order to break the alignment between
    # the ID/Edge ordering and the order of creation/addition
    node11 = core.Node(ID="1.1", root=p, tag="1")
    node13 = core.Node(ID="1.3", root=p, tag="3", attrib={"node": True})
    node12 = core.Node(ID="1.2", root=p, tag="x",
                       orderkey=operator.attrgetter("tag"))
    node21 = core.Node(ID="2.1", root=p, tag="2")
    node22 = core.Node(ID="2.2", root=p, tag="2")
    node12.add("test2", node11)
    node12.add("test1", node13, edge_attrib={"edge": True})
    node21.add("test2", node12)
    node21.add("test", node11)
    node22.add("test1", node12)
    node22.add("test", node13)
    node22.add("test", node11)
    return p


def l1_passage():
    """Creates a Passage to work with using layer1 objects.

    Annotation layout (what annotation each terminal has):
        1: Linker, linked with the first parallel scene
        2-10: Parallel scene #1, 2-5 ==> Participant #1
            6-9 ==> Process #1, 10 ==> Punctuation, remote Participant is
            Adverbial #2
        11-19: Parallel scene #23, which encapsulated 2 scenes and a linker
            (not a real scene, has no process, only for grouping)
        11-15: Parallel scene #2 (under #23), 11-14 ==> Participant #3,
            15 ==> Adverbial #2, remote Process is Process #1
        16: Linker #2, links Parallel scenes #2 and #3
        17-19: Parallel scene #3, 17-18 ==> Process #3,
            19 ==> Participant #3, implicit Participant
        20: Punctuation (under the head)

    """

    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 20 terminals (1-20), #10 and #20 are punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0))
             for i in range(1, 21)]

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
    # Scene #2: [[11 12 13 14 P] [15 D]]
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


def multi_sent():
    """Creates a :class:`Passage` with multiple sentences and paragraphs.

    Passage: [1 2 [3 P] H] . [[5 6 . P] H]
             [[8 P] . 10 . H]

    """
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    terms = [l0.add_terminal(str(i), False) for i in range(1, 4)]
    terms.append(l0.add_terminal(".", True))
    terms.append(l0.add_terminal("5", False))
    terms.append(l0.add_terminal("6", False))
    terms.append(l0.add_terminal(".", True))
    terms.append(l0.add_terminal("8", False, paragraph=2))
    terms.append(l0.add_terminal(".", True, paragraph=2))
    terms.append(l0.add_terminal("10", False, paragraph=2))
    terms.append(l0.add_terminal(".", True, paragraph=2))
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    h2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    h3 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
    p2 = l1.add_fnode(h2, layer1.EdgeTags.Process)
    p3 = l1.add_fnode(h3, layer1.EdgeTags.Process)
    h1.add(layer1.EdgeTags.Terminal, terms[0])
    h1.add(layer1.EdgeTags.Terminal, terms[1])
    p1.add(layer1.EdgeTags.Terminal, terms[2])
    l1.add_punct(None, terms[3])
    p2.add(layer1.EdgeTags.Terminal, terms[4])
    p2.add(layer1.EdgeTags.Terminal, terms[5])
    l1.add_punct(p2, terms[6])
    p3.add(layer1.EdgeTags.Terminal, terms[7])
    l1.add_punct(h3, terms[8])
    h3.add(layer1.EdgeTags.Terminal, terms[9])
    l1.add_punct(h3, terms[10])
    return p


def crossing():
    """Creates a :class:`Passage` with multiple sentences and paragraphs, with crossing edges.

    Passage: [1 2 [3 P(remote)] H] .
             [[3 P] . 4 . H]

    """
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    terms = [
        l0.add_terminal("1", False),
        l0.add_terminal("2", False),
        l0.add_terminal(".", True),
        l0.add_terminal("3", False, paragraph=2),
        l0.add_terminal(".", True, paragraph=2),
        l0.add_terminal("4", False, paragraph=2),
        l0.add_terminal(".", True, paragraph=2),
    ]
    h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    h2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    p1 = l1.add_fnode(h2, layer1.EdgeTags.Process)
    l1.add_remote(h1, layer1.EdgeTags.Process, p1)
    h1.add(layer1.EdgeTags.Terminal, terms[0])
    h1.add(layer1.EdgeTags.Terminal, terms[1])
    l1.add_punct(None, terms[2])
    p1.add(layer1.EdgeTags.Terminal, terms[3])
    l1.add_punct(h2, terms[4])
    h2.add(layer1.EdgeTags.Terminal, terms[5])
    l1.add_punct(h2, terms[6])
    return p


def discontiguous():
    """Creates a highly-discontiguous Passage object."""
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    # 20 terminals (1-20), #10 and #20 are punctuation
    terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0))
             for i in range(1, 21)]

    # First parallel scene, stretching on terminals 1-10
    # The dashed edge tags (e.g. -C, C-) mean discontiguous units
    # [PS [D [E 0] [C- 1] [E 2] [-C 3]]
    #     [A- 4] [P- 5 6] [-A 7] [F 8] [-P [U 9]]]
    # In addition, D takes P as a remote G
    ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    d1 = l1.add_fnode(ps1, layer1.EdgeTags.Adverbial)
    e1 = l1.add_fnode(d1, layer1.EdgeTags.Elaborator)
    c1 = l1.add_fnode(d1, layer1.EdgeTags.Center)
    e2 = l1.add_fnode(d1, layer1.EdgeTags.Elaborator)
    a1 = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
    p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
    f1 = l1.add_fnode(ps1, layer1.EdgeTags.Function)
    l1.add_remote(d1, layer1.EdgeTags.Ground, p1)
    e1.add(layer1.EdgeTags.Terminal, terms[0])
    c1.add(layer1.EdgeTags.Terminal, terms[1])
    e2.add(layer1.EdgeTags.Terminal, terms[2])
    c1.add(layer1.EdgeTags.Terminal, terms[3])
    a1.add(layer1.EdgeTags.Terminal, terms[4])
    p1.add(layer1.EdgeTags.Terminal, terms[5])
    p1.add(layer1.EdgeTags.Terminal, terms[6])
    a1.add(layer1.EdgeTags.Terminal, terms[7])
    f1.add(layer1.EdgeTags.Terminal, terms[8])
    l1.add_punct(p1, terms[9])

    # Second parallel scene, stretching on terminals 11-14 + 18-20
    # [PS- [D IMPLICIT] [G IMPLICIT] [P 10 11 12 13]]
    # [-PS [A 17 18 [U 19]]]
    ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    l1.add_fnode(ps2, layer1.EdgeTags.Adverbial, implicit=True)
    l1.add_fnode(ps2, layer1.EdgeTags.Ground, implicit=True)
    p2 = l1.add_fnode(ps2, layer1.EdgeTags.Process)
    a2 = l1.add_fnode(ps2, layer1.EdgeTags.Participant)
    p2.add(layer1.EdgeTags.Terminal, terms[10])
    p2.add(layer1.EdgeTags.Terminal, terms[11])
    p2.add(layer1.EdgeTags.Terminal, terms[12])
    p2.add(layer1.EdgeTags.Terminal, terms[13])
    a2.add(layer1.EdgeTags.Terminal, terms[17])
    a2.add(layer1.EdgeTags.Terminal, terms[18])
    l1.add_punct(a2, terms[19])

    # Third parallel scene, stretching on terminals 15-17
    # [PS [P IMPLICIT] 14 [A 15 16]]
    ps3 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
    ps3.add(layer1.EdgeTags.Terminal, terms[14])
    l1.add_fnode(ps3, layer1.EdgeTags.Process, implicit=True)
    a3 = l1.add_fnode(ps3, layer1.EdgeTags.Participant)
    a3.add(layer1.EdgeTags.Terminal, terms[15])
    a3.add(layer1.EdgeTags.Terminal, terms[16])

    return p


def loaded():
    return convert.from_standard(load_xml("test_files/standard3.xml"))


def loaded_valid():
    return convert.from_standard(load_xml("test_files/standard3_valid.xml"))


def load_xml(path):
    """XML file path ==> root element
    :param path: path to XML file
    """
    with open(path, encoding="utf-8") as f:
        return ETree.ElementTree().parse(f)


PASSAGES = (loaded, loaded_valid, multi_sent, crossing, discontiguous, l1_passage, empty)


def create_passage(num_terms=3, *punct):
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    terms = [l0.add_terminal(text=str(i), punct=(i in punct)) for i in range(1, num_terms + 1)]
    return p, l1, terms


def attach_terminals(terms, *nodes):
    for term, node in zip(terms, nodes):
        node.add(layer1.EdgeTags.Terminal, term)
