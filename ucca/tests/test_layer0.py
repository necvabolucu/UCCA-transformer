from ucca import core, layer0

"""Tests module layer0 functionality."""


def test_terminals():
    """Tests :class:`layer0`.Terminal new and inherited functionality."""
    p = core.Passage("1")
    layer0.Layer0(p)
    terms = [
        layer0.Terminal(ID="0.1", root=p,
                        tag=layer0.NodeTags.Word,
                        attrib={"text": "1",
                                "paragraph": 1,
                                "paragraph_position": 1}),
        layer0.Terminal(ID="0.2", root=p,
                        tag=layer0.NodeTags.Word,
                        attrib={"text": "2",
                                "paragraph": 2,
                                "paragraph_position": 1}),
        layer0.Terminal(ID="0.3", root=p,
                        tag=layer0.NodeTags.Punct,
                        attrib={"text": ".",
                                "paragraph": 2,
                                "paragraph_position": 2})
    ]

    p_copy = core.Passage("2")
    layer0.Layer0(p_copy)
    equal_term = layer0.Terminal(ID="0.1", root=p_copy,
                                 tag=layer0.NodeTags.Word,
                                 attrib={"text": "1",
                                         "paragraph": 1,
                                         "paragraph_position": 1})
    unequal_term = layer0.Terminal(ID="0.2", root=p_copy,
                                   tag=layer0.NodeTags.Word,
                                   attrib={"text": "two",
                                           "paragraph": 2,
                                           "paragraph_position": 1})

    assert [t.punct for t in terms] == [False, False, True]
    assert [t.text for t in terms] == ["1", "2", "."]
    assert [t.position for t in terms] == [1, 2, 3]
    assert [t.paragraph for t in terms] == [1, 2, 2]
    assert [t.para_pos for t in terms] == [1, 1, 2]
    assert not (terms[0] == terms[1])
    assert not (terms[0] == terms[2])
    assert not (terms[1] == terms[2])
    assert terms[0] == terms[0]
    assert terms[0].equals(equal_term)
    assert not (terms[1].equals(unequal_term))


def test_layer0():
    p = core.Passage("1")
    l0 = layer0.Layer0(p)
    t1 = l0.add_terminal(text="1", punct=False)
    l0.add_terminal(text="2", punct=True, paragraph=2)
    t3 = l0.add_terminal(text="3", punct=False, paragraph=2)
    assert [x[0] for x in l0.pairs] == [1, 2, 3]
    assert [t.para_pos for t in l0.all] == [1, 1, 2]
    assert l0.words == (t1, t3)
