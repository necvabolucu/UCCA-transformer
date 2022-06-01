import xml.etree.ElementTree as ETree

from ucca import layer0, layer1, convert, textutil
from .conftest import loaded, load_xml

"""Tests convert module correctness and API."""


def _test_edges(node, tags):
    """Tests that the node edge tags and number match tags argument."""
    assert len(node) == len(tags)
    for edge, tag in zip(node, tags):
        assert edge.tag == tag


def _test_terms(node, terms):
    """Tests that node contain the terms given, and only them."""
    for edge, term in zip(node, terms):
        assert edge.tag == layer1.EdgeTags.Terminal
        assert edge.child == term


def test_site_terminals():
    elem = load_xml("test_files/site1.xml")
    passage = convert.from_site(elem)
    terms = passage.layer(layer0.LAYER_ID).all

    assert passage.ID == "118"
    assert len(terms) == 15

    # There are two punctuation signs (dots, positions 5 and 11), which
    # also serve as paragraph end points. All others are words whose text
    # is their positions, so test that both text, punctuation (yes/no)
    # and paragraphs are converted correctly
    for i, t in enumerate(terms):
        # i starts in 0, positions at 1, hence 5,11 ==> 4,10
        if i in (4, 10):
            assert t.text == "." and t.punct
        else:
            assert t.text == str(i + 1) and not t.punct
        if i < 5:
            par = 1
        elif i < 11:
            par = 2
        else:
            par = 3
        assert t.paragraph == par


def test_site_simple():
    elem = load_xml("test_files/site2.xml")
    passage = convert.from_site(elem)
    terms = passage.layer(layer0.LAYER_ID).all
    l1 = passage.layer("1")

    # The Terminals in the passage are just like in test_site_terminals,
    # with this layer1 hierarchy: [[1 C] [2 E] L] [3 4 . H]
    # with the linker having a remark and the parallel scene is uncertain
    head = l1.heads[0]
    assert len(head) == 12  # including all "unused" terminals
    assert head[9].tag == layer1.EdgeTags.Linker
    assert head[10].tag == layer1.EdgeTags.ParallelScene
    linker = head.children[9]
    _test_edges(linker, [layer1.EdgeTags.Center,
                         layer1.EdgeTags.Elaborator])
    assert linker.extra["remarks"], 'remark'
    center = linker.children[0]
    elab = linker.children[1]
    _test_terms(center, terms[0:1])
    _test_terms(elab, terms[1:2])
    ps = head.children[10]
    _test_edges(ps, [layer1.EdgeTags.Terminal,
                     layer1.EdgeTags.Terminal,
                     layer1.EdgeTags.Punctuation])
    assert ps.attrib.get("uncertain")
    assert ps.children[0] == terms[2]
    assert ps.children[1] == terms[3]
    assert ps.children[2].children[0] == terms[4]


def test_site_advanced():
    elem = load_xml("test_files/site3.xml")
    passage = convert.from_site(elem)
    terms = passage.layer(layer0.LAYER_ID).all
    l1 = passage.layer("1")

    # This passage has the same terminals as the simple and terminals test,
    # and have the same layer1 units for the first paragraph as the simple
    # test. In addition, it has the following annotation:
    # [6 7 8 9 H] [10 F] .
    # the 6-9 H has remote D which is [10 F]. Inside of 6-9, we have [8 S]
    # and [6 7 ... 9 A], where [6 E] and [7 ... 9 C].
    # [12 H] [13 H] [14 H] [15 L], where 15 linkage links 12, 13 and 14 and
    # [15 L] has an implicit Center unit
    head, lkg = l1.heads
    _test_edges(head, [layer1.EdgeTags.Linker,
                       layer1.EdgeTags.ParallelScene,
                       layer1.EdgeTags.ParallelScene,
                       layer1.EdgeTags.Function,
                       layer1.EdgeTags.Punctuation,
                       layer1.EdgeTags.ParallelScene,
                       layer1.EdgeTags.ParallelScene,
                       layer1.EdgeTags.ParallelScene,
                       layer1.EdgeTags.Linker])

    # we only take what we haven"t checked already
    ps1, func, punct, ps2, ps3, ps4, link = head.children[2:]
    _test_edges(ps1, [layer1.EdgeTags.Participant,
                      layer1.EdgeTags.Process,
                      layer1.EdgeTags.Adverbial])
    assert ps1[2].attrib.get("remote")
    ps1_a, ps1_p, ps1_d = ps1.children
    _test_edges(ps1_a, [layer1.EdgeTags.Elaborator,
                        layer1.EdgeTags.Center])
    _test_terms(ps1_a.children[0], terms[5:6])
    _test_terms(ps1_a.children[1], terms[6:9:2])
    _test_terms(ps1_p, terms[7:8])
    assert ps1_d == func
    _test_terms(func, terms[9:10])
    _test_terms(punct, terms[10:11])
    _test_terms(ps2, terms[11:12])
    _test_terms(ps3, terms[12:13])
    _test_terms(ps4, terms[13:14])
    assert len(link) == 2
    assert link[0].tag == layer1.EdgeTags.Center
    assert link.children[0].attrib.get("implicit")
    assert link[1].tag == layer1.EdgeTags.Elaborator
    assert link.children[1][0].tag == layer1.EdgeTags.Terminal
    assert link.children[1][0].child == terms[14]
    assert lkg.relation == link
    assert lkg.arguments == [ps2, ps3, ps4]


def test_site_discontiguous_with_remote():
    elem = load_xml("test_files/site4.xml")
    passage = convert.from_site(elem)
    s1 = passage.layer(layer1.LAYER_ID).heads[0].state
    remote_a1 = [e.child for e in s1 if e.attrib.get("remote") and e.tag == layer1.EdgeTags.Participant]
    assert len(remote_a1) == 1
    a1 = remote_a1[0]
    remote_a2 = [e.child for e in a1 if e.attrib.get("remote") and e.tag == layer1.EdgeTags.Participant]
    assert len(remote_a2) == 1


def test_site_discontiguous_with_implicit():
    elem = load_xml("test_files/site5.xml")
    passage = convert.from_site(elem)
    s1 = passage.layer(layer1.LAYER_ID).heads[0].state
    remote_t1 = [e.child for e in s1 if e.child.attrib.get("implicit") and e.tag == layer1.EdgeTags.Time]
    assert len(remote_t1) == 1


def test_to_standard():
    passage = convert.from_site(load_xml("test_files/site3.xml"))
    ref = load_xml("test_files/standard3.xml")  # old format of xml
    new_ref = convert.to_standard(convert.from_standard(ref))   # converting to the new xml format
    root = convert.to_standard(passage)
    assert textutil.indent_xml(ETree.tostring(new_ref)).splitlines() == \
            textutil.indent_xml(ETree.tostring(root)).splitlines()


def test_from_standard():
    passage = loaded()
    ref = convert.from_site(load_xml("test_files/site3.xml"))
    assert passage.equals(ref, ordered=True)


def test_from_text():
    sample = ["Hello . again", "nice", " ? ! end", ""]
    passage = next(convert.from_text(sample))
    terms = passage.layer(layer0.LAYER_ID).all
    pos = 0
    for i, par in enumerate(sample):
        for text in par.split():
            assert terms[pos].text == text
            assert terms[pos].paragraph == i + 1
            pos += 1


def test_from_text_long():
    sample = """
        After graduation, John moved to New York City.

        He liked it there. He played tennis.
        And basketball.

        And he lived happily ever after.
        """
    passages = list(convert.from_text(sample))
    assert len(passages) == 3, list(map(convert.to_text, passages))


def test_to_text():
    passage = loaded()
    assert convert.to_text(passage, False)[0] == "1 2 3 4 . 6 7 8 9 10 . 12 13 14 15"
    assert convert.to_text(passage, True) == ["1 2 3 4 .", "6 7 8 9 10 .", "12 13 14 15"]


def test_to_site():
    passage = loaded()
    root = convert.to_site(passage)
    copy = convert.from_site(root)
    assert passage.equals(copy)
