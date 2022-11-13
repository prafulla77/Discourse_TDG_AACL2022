ROOT_word = 'root'
ROOT_label = 'ROOT'


class Node:
    def __init__(self, snt_index_in_doc=-1, start_word_index_in_snt=-1, end_word_index_in_snt=-1, node_index_in_doc=-1,
                 start_word_index_in_doc=-1, end_word_index_in_doc=-1, words=ROOT_word, label=ROOT_label):
        self.snt_index_in_doc = snt_index_in_doc
        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt
        self.node_index_in_doc = node_index_in_doc
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc

        self.words = words
        self.full_label = label  # full label
        self.TE_label = label.split('-')[0].upper()  # timex/event label

        self.ID = '_'.join([str(snt_index_in_doc),
                            str(start_word_index_in_snt), str(end_word_index_in_snt)])

    def __str__(self):
        return '\t'.join([self.ID, self.words, self.full_label])

    def __eq__(self, other):
        if isinstance(other, Node):
            # Note that this does not check whether the two nodes are from the same document
            return self.ID == other.ID
        return False

    def __hash__(self):
        return hash(self.ID)


def get_root_node():
    return Node()


def is_root_node(node):
    return node.node_index_in_doc == -1


EDGE_LABEL_LIST = [
    'EE-before',
    'EE-Depend-on',
    'EE-after',
    'EE-overlap',
    'ET-before',
    'ET-after',
    'ET-overlap',
    'ET-included',
    'TT-Depend-on',
    'TT-included'
]


EDGE_LABEL_COMPRESSED = [
    'before',
    'Depend-on',
    'after',
    'overlap',
    'included'
]


DP_TAGS = [
    '#1#',
    '#2#',
    '#3#',
    '#4#',
    '#5#',
    '#6#',
    '#7#',
    '#8#',
    '#9#',
]


# 1 for same type, 2 for before, 3 for after, 4 everything else
DP_MAP = {
    (1, 1): 1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (5, 5): 1, (6, 6): 1, (7, 7): 1, (8, 8): 1,
    (1, 2): 2, (3, 1): 2, (5, 1): 2, (1, 8): 2,
    (2, 1): 3, (1, 3): 3, (1, 5): 3, (8, 1): 3,
    (2, 3): 3, (2, 5): 3,
    (3, 2): 2, (5, 2): 2,
    (3, 5): 3, (8, 3): 3,
    (5, 3): 2, (3, 8): 2,
    (4, 5): 3,
    (5, 4): 2,
    (5, 7): 2, (5, 8): 2,
    (7, 5): 3, (8, 5): 3,
    (7, 8): 2,
    (8, 7): 3
}
