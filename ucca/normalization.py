from ucca import layer0, layer1
from ucca.layer0 import NodeTags as L0Tags
from ucca.layer1 import EdgeTags as ETags, NodeTags as L1Tags


def fparent(node_or_edge):
    try:
        return node_or_edge.fparent
    except AttributeError:
        try:
            return node_or_edge.parent
        except AttributeError:
            return node_or_edge.parents[0] if node_or_edge.parents else None


def remove_unmarked_implicits(node):
    while node is not None and not node.children and not node.attrib.get("implicit"):
        parent = fparent(node)
        if parent is None:
            break
        node.destroy()
        node = parent


def remove(parent, child):
    if parent is not None:
        parent.remove(child)
        remove_unmarked_implicits(parent)


def destroy(node_or_edge):
    parent = fparent(node_or_edge)
    try:
        node_or_edge.destroy()
    except AttributeError:
        parent.remove(node_or_edge)
    if parent is not None:
        remove_unmarked_implicits(parent)
    return parent


def copy_edge(edge, parent=None, child=None, tag=None, attrib=None):
    if parent is None:
        parent = edge.parent
    if child is None:
        child = edge.child
    if not tag:
        categories = [(c.tag, c.slot, c.layer, c.parent) for c in edge.categories]
    else:
        categories = [(tag,)]
    if attrib is None:
        attrib = edge.attrib
    if parent in child.iter():
        # raise ValueError("Created cycle (%s->%s) when trying to normalize '%s'" % (
        #     "->".join(n.ID for n in child.iter() if parent in n.iter()), child.ID, parent))
        return False
    parent.add_multiple(categories, child, edge_attrib=attrib)
    return True


def replace_center(edge):
    if len(edge.parent) == 1 and not edge.parent.parents:
        return ETags.ParallelScene
    if edge.parent.participants and not edge.parent.is_scene():
        return ETags.Process  # TODO should be state if the word is a copula
    return edge.tag


def replace_edge_tags(node):
    for edge in node:
        if not edge.attrib.get("remote") and edge.tag == ETags.Center:
            edge.tag = replace_center(edge)
        elif node.parallel_scenes:
            if edge.tag == ETags.Connector:
                edge.tag = ETags.Linker
        elif edge.tag == ETags.Linker:
            edge.tag = ETags.Connector
        elif node.is_scene():
            if edge.tag == ETags.Elaborator:
                edge.tag = ETags.Adverbial
        elif edge.tag == ETags.Adverbial:
            edge.tag = ETags.Elaborator


def move_elements(node, tags, parent_tags, forward=True):
    for edge in node:
        if edge.child.tag == L1Tags.Foundational and edge.tag in ((tags,) if isinstance(tags, str) else tags):
            try:
                parent_edge = min((e for e in node if e != edge and e.child.tag == L1Tags.Foundational),
                                  key=lambda e: abs(((edge.child.start_position - e.child.end_position),
                                                     (e.child.start_position - edge.child.end_position))[forward]))
            except ValueError:
                continue
            if parent_edge.tag in ((parent_tags,) if isinstance(parent_tags, str) else parent_tags):
                parent = parent_edge.child
                if copy_edge(edge, parent=parent):
                    remove(node, edge)


def move_scene_elements(node):
    if node.parallel_scenes:
        move_elements(node, tags=(ETags.Relator, ETags.Elaborator, ETags.Center), parent_tags=ETags.ParallelScene)


def move_sub_scene_elements(node):
    if node.is_scene():
        move_elements(node, tags=(ETags.Elaborator, ETags.Center), parent_tags=ETags.Participant, forward=False)


def separate_scenes(node, l1, top_level=False):
    if (node.is_scene() or node.participants) and (top_level or node.parallel_scenes):
        edges = list(node)
        scene = l1.add_fnode(node, ETags.ParallelScene)
        for edge in edges:
            if edge.tag not in (ETags.ParallelScene, ETags.Punctuation, ETags.Linker, ETags.Ground):
                if copy_edge(edge, parent=scene):
                    remove(node, edge)


def lowest_common_ancestor(*nodes):
    parents = [nodes[0]] if nodes else []
    while parents:
        for parent in parents:
            if parent.tag == L1Tags.Foundational and (not parent.terminals or nodes[1:]) \
                    and all(n in parent.iter() for n in nodes[1:]):
                return parent
        parents = [p for n in parents for p in n.parents]
    return None


def nearest_word(l0, position, step):
    while True:
        position += step
        try:
            terminal = l0.by_position(position)
        except IndexError:
            return None
        if terminal.tag == L0Tags.Word:
            return terminal


def nearest_parent(l0, *terminals):
    return lowest_common_ancestor(*filter(None, (nearest_word(l0, terminals[0].position, -1),
                                                 nearest_word(l0, terminals[-1].position, 1))))


def reattach_punct(l0, l1):
    detach_punct(l1)
    attach_punct(l0, l1)


def attach_punct(l0, l1):
    for terminal in l0.all:
        if layer0.is_punct(terminal) and not terminal.incoming:
            l1.add_punct(nearest_parent(l0, terminal), terminal)


def detach_punct(l1):
    for node in l1.all:
        if node.tag == L1Tags.Punctuation:
            destroy(node)


def reattach_terminals(l0, l1):
    attach_terminals(l0, l1)
    for terminal in l0.all:
        for edge in terminal.incoming:
            if any(e.tag != ETags.Terminal for e in edge.parent):
                node = l1.add_fnode(edge.parent, layer1.EdgeTags.Center)
                if copy_edge(edge, parent=node):
                    remove(edge.parent, edge)


def attach_terminals(l0, l1):
    for terminal in l0.all:
        if not terminal.incoming:
            node = l1.add_fnode(nearest_parent(l0, terminal), layer1.EdgeTags.Function)
            node.add(layer1.EdgeTags.Terminal, terminal)


def flatten_centers(node):
    """
    Whenever there are Cs inside Cs, remove the external C.
    Whenever there is a C as an only child, remove it.
    """
    if node.tag == L1Tags.Foundational and len(node.centers) == 1:
        if node.ftag == ETags.Center and len(fparent(node).centers) == 1:  # Center inside center
            for edge in node.incoming:
                if edge.attrib.get("remote"):
                    copy_edge(edge, child=node.centers[0])
            for edge in node:
                copy_edge(edge, parent=fparent(node))
            return destroy(node)
        elif len(node.children) == 1:  # Center as only child
            for edge in node.incoming:
                attrib = edge.attrib
                if node.outgoing[0].attrib.get("remote"):
                    attrib["remote"] = True
                copy_edge(edge, child=node.centers[0], attrib=attrib)
            return destroy(node)
    return node


def flatten_functions(node):
    """
    Whenever there is an F as an only child, remove it. If an F has non-terminal children, move them up.
    """
    if node.tag == L1Tags.Foundational and node.incoming:  # Avoid creating root->terminal edge
        for child in node.functions:
            if len(child.children) > len(child.terminals):
                for edge in child:
                    copy_edge(edge, parent=node, tag = ETags.Function if edge.tag == ETags.Center else edge.tag)
                destroy(child)
        if len(node.functions) == len(node.children) == 1:
            for edge in node.incoming:
                copy_edge(edge, child=node.functions[0])
            return destroy(node)
    return node


def flatten_participants(node):
    """
    Whenever there is an A as an only child, remove it.
    If there is an implicit A in a scene without a main relation, remove it.
    """
    if node.tag == L1Tags.Foundational:
        participants = node.participants
        if len(participants) == len(node.children) == 1 and len(participants[0].ftags) == 1:
            for edge in node.incoming:
                copy_edge(edge, child=participants[0])
            return destroy(node)
        elif participants and not node.is_scene():
            for child in participants:
                if child.attrib.get("implicit"):
                    destroy(child)
    return node


def normalize_node(node, l1, extra):
    if node.tag == L1Tags.Foundational:
        if extra:
            replace_edge_tags(node)
            move_scene_elements(node)
            move_sub_scene_elements(node)
        separate_scenes(node, l1, top_level=node in l1.heads)
        node = flatten_centers(node)
        if node is None:
            return
        node = flatten_functions(node)
        if node is None:
            return
        flatten_participants(node)


def normalize(passage, extra=False):
    l0 = passage.layer(layer0.LAYER_ID)
    l1 = passage.layer(layer1.LAYER_ID)
    reattach_punct(l0, l1)
    heads = list(l1.heads)
    stack = [heads]
    visited = set()
    path = []
    path_set = set()
    while stack:
        for edge in stack[-1]:
            try:
                node = edge.child
            except AttributeError:
                node = edge
            if node in path_set:
                destroy(edge)
            elif node not in visited:
                visited.add(node)
                path.append(node)
                path_set.add(node)
                stack.append(node)
                normalize_node(node, l1, extra)
                break
        else:
            if path:
                path_set.remove(path.pop())
            stack.pop()
    reattach_punct(l0, l1)
    if extra:
        reattach_terminals(l0, l1)
