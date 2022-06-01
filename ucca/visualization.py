import re
import warnings
from collections import defaultdict
from operator import attrgetter

from ucca import layer0, layer1


def node_label(node):
    return re.sub("[^(]*\((.*)\)", "\\1", node.attrib.get("label", ""))


def draw(passage, node_ids=False):
    import matplotlib.cbook
    import networkx as nx
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    warnings.filterwarnings("ignore", category=UserWarning)
    g = nx.DiGraph()
    terminals = sorted(passage.layer(layer0.LAYER_ID).all, key=attrgetter("position"))
    g.add_nodes_from([(n.ID, {"label": n.text, "color": "white"}) for n in terminals])
    g.add_nodes_from([(n.ID, {"label": node_label(n) or ("", "IMPLICIT", n.ID)[n.attrib.get("implicit", 2 * node_ids)],
                              "color": ("black", "gray", "white")[n.tag == layer1.NodeTags.Linkage or
                                                                  2 * n.attrib.get("implicit", node_ids)]})
                      for n in passage.layer(layer1.LAYER_ID).all])
    g.add_edges_from([(n.ID, e.child.ID, {"label": "|".join(e.tags),
                                          "style": "dashed" if e.attrib.get("remote") else "solid"})
                      for layer in passage.layers for n in layer.all for e in n])
    pos = topological_layout(passage)
    nx.draw(g, pos, arrows=False, font_size=10,
            node_color=[d["color"] for _, d in g.nodes(data=True)],
            labels={n: d["label"] for n, d in g.nodes(data=True) if d["label"]},
            style=[d["style"] for _, _, d in g.edges(data=True)])
    nx.draw_networkx_edge_labels(g, pos, font_size=8,
                                 edge_labels={(u, v): d["label"] for u, v, d in g.edges(data=True)})


def topological_layout(passage):
    visited = defaultdict(set)
    pos = {}
    implicit_offset = 1 + max((n.position for n in passage.layer(layer0.LAYER_ID).all), default=-1)
    remaining = [n for layer in passage.layers for n in layer.all if not n.parents]
    while remaining:
        node = remaining.pop()
        if node.ID in pos:  # done already
            continue
        if node.children:
            children = [c for c in node.children if c.ID not in pos and c not in visited[node.ID]]
            if children:
                visited[node.ID].update(children)  # to avoid cycles
                remaining += [node] + children
                continue
            xs, ys = zip(*(pos[c.ID] for c in node.children))
            pos[node.ID] = (sum(xs) / len(xs), 1 + max(ys) ** 1.01)  # done with children
        elif node.layer.ID == layer0.LAYER_ID:  # terminal
            pos[node.ID] = (int(node.position), 0)
        else:  # implicit
            pos[node.ID] = (implicit_offset, 0)
            implicit_offset += 1
    return pos


TEX_ESCAPE_TABLE = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
    "\\": r"\textbackslash{}",
    "<": r"\textless ",
    ">": r"\textgreater ",
}
TEX_ESCAPE_PATTERN = re.compile("|".join(map(re.escape, sorted(TEX_ESCAPE_TABLE, key=len, reverse=True))))


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    return TEX_ESCAPE_PATTERN.sub(lambda match: TEX_ESCAPE_TABLE[match.group()], text)


def tikz(p, indent=None, node_ids=False):
    # child {node (After) [word] {After} edge from parent node[above] {\scriptsize $L$}}
    # child {node (graduation) [circle] {}
    # {
    # child {node [word] {graduation} edge from parent node[left] {\scriptsize $P$}}
    # } edge from parent node[right] {\scriptsize $H$} }
    # child {node [word] {,} edge from parent node[below] {\scriptsize $U$}}
    # child {node (moved) [circle] {}
    # {
    # child {node (John) [word] {John} edge from parent node[left] {\scriptsize $A$}}
    # child {node [word] {moved} edge from parent node[left] {\scriptsize $P$}}
    # child {node [circle] {}
    # {
    # child {node [word] {to} edge from parent node[left] {\scriptsize $R$}}
    # child {node [word] {Paris} edge from parent node[right] {\scriptsize $C$}}
    # } edge from parent node[right] {\scriptsize $A$} }
    # } edge from parent node[right] {\scriptsize $H$} }
    # ;
    # \draw[dashed,->] (graduation) to node [auto] {\scriptsize $A$} (John);
    if indent is None:
        l1 = p.layer(layer1.LAYER_ID)
        return r"""
\begin{tikzpicture}[->,level distance=1cm,
  level 1/.style={sibling distance=4cm},
  level 2/.style={sibling distance=15mm},
  level 3/.style={sibling distance=15mm},
  every circle node/.append style={%s=black}]
  \tikzstyle{word} = [font=\rmfamily,color=black]
  """ % ("draw" if node_ids else "fill") + "\\" + tikz(l1.heads[0], indent=1, node_ids=node_ids) + \
            "\n".join([";"] + ["  \draw[dashed,->] (%s) to node [auto] {\scriptsize $%s$} (%s);" %
                               (e.parent.ID.replace(".", "_"), "|".join(e.tags), e.child.ID.replace(".", "_"))
                               for n in l1.all for e in n if e.attrib.get("remote")] + [r"\end{tikzpicture}"])
    return "node (" + p.ID.replace(".", "_") + ") " + (
        ("[word] {" +
         (" ".join(tex_escape(t.text)
                   for t in sorted(p.terminals, key=attrgetter("position"))) or r"\textbf{IMPLICIT}")
         + "} ") if p.terminals or p.attrib.get("implicit") else ("\n" + indent * "  ").join(
            ["[circle] {%s}" % (node_label(p) or (p.ID if node_ids else "")), "{"] +
            ["child {" + tikz(e.child, indent + 1) +
             " edge from parent node[auto]  {\scriptsize $" + "|".join(e.tags) + "$}}"
             for e in sorted(p, key=lambda f: f.child.start_position)
             if not e.attrib.get("remote")] +
            ["}"]))
