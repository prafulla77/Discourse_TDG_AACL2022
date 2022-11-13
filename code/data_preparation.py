import codecs
import collections

from data_structures import Node, get_root_node, ROOT_label


def create_snt_edge_lists(doc):
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    return snt_list, edge_list


def create_node_dict(snt_list, edge_list):
    node_dict = {}
    node_label = {'-1_-1_-1': ROOT_label}
    event_parents = collections.defaultdict(list)
    for i, edge in enumerate(edge_list):
        child, c_label, parent, _ = edge  # parent and edge label aren't used here
        if child not in node_label:
            c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]
            node_label[child] = c_label
            c_node = Node(c_snt, c_start, c_end, i,
                          get_word_index_in_doc(snt_list, c_snt, c_start),
                          get_word_index_in_doc(snt_list, c_snt, c_end),
                          ' '.join(snt_list[c_snt][c_start:c_end + 1]),
                          c_label)
            # node_dict.append(c_node)
            node_dict[child] = c_node

        # Track both Timex and Event parents for Event Nodes
        if c_label == 'Event':
            event_parents[child].append(i)

    node_type = ['Timex']*len(edge_list)

    for event in event_parents:
        # Ensure all event nodes have two parents
        assert len(event_parents[event]) == 2

    for i, edge in enumerate(edge_list):
        child, c_label, parent, _ = edge
        if c_label == 'Event':
            first, second = event_parents[child]
            first_parent, second_parent = edge_list[first][2], edge_list[second][2]
            if node_label[first_parent] == ROOT_label and node_label[second_parent] == ROOT_label:
                node_type[second] = 'Event'
            else:
                if node_label[first_parent] == 'Timex':
                    node_type[second] = 'Event'
                elif node_label[second_parent] == 'Timex':
                    node_type[first] = 'Event'
    return node_dict, node_type


def get_word_index_in_doc(snt_list, snt_index_in_doc, word_index_in_snt):
    index = 0
    for i, snt in enumerate(snt_list):
        if i < snt_index_in_doc:
            index += len(snt)
        else:
            break

    return index + word_index_in_snt


def check_example_contains_gold_parent(example):
    for tup in example:
        if tup[2] != 'NO_EDGE':
            return True

    return False


def get_discourse_tags(folder_name, doc_name):
    dp_file = open('../tdg_data/'+folder_name+'_dp/'+doc_name)
    dp_tags = [0]  # For the first node, it's mostly DCT
    for index, line in enumerate(dp_file):
        if index == 0:
            continue
        score = [float(elem) for elem in line.strip().split('\t')[1:]]
        idx = score.index(max(score))
        dp_tags.append(idx+1)  # DP_TAGS[idx] # Adding 1 to offset the first sentence
    # dp_tags.append(10)  # for the root node, sentence id = -1
    return dp_tags


def make_one_doc_training_data(doc, folder_name):
    """
    return: training_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'before'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    doc_name = doc[0].split('doc id="')[1].split('"')[0]
    snt_list, edge_list = create_snt_edge_lists(doc)
    dp_tags = get_discourse_tags(folder_name, doc_name)

    # create node_dict
    node_dict, node_type = create_node_dict(snt_list, edge_list)

    # create training example list
    training_example_list = []
    root_node = get_root_node()

    for i, edge in enumerate(edge_list):
        child, _, parent, label = edge
        child_node = node_dict[child]

        example = choose_candidates(node_dict, root_node, child_node, parent, label, node_type[i])

        if check_example_contains_gold_parent(example):
            training_example_list.append(example)
        else:
            # Either child == parent in the annotation (can happen for Turker annotation) which is invalid
            doc_name = doc[0].split(':')[1]
            raise ValueError('No gold parent for edge {} in document {}'.format(edge, doc_name))

    return [snt_list, training_example_list, dp_tags]


def choose_candidates(node_dict, root_node, child, parent_ID, label, node_label):
    candidates = []

    # Always consider root
    candidates.append(get_candidate(child, root_node, parent_ID, label))

    for candidate_key in node_dict:
        candidate_node = node_dict[candidate_key]
        if candidate_node.full_label != node_label:
            continue
        if candidate_node.ID == child.ID:
            continue
        # elif candidate_node.snt_index_in_doc - child.snt_index_in_doc > 2:
        #     # Only consider from beginning of text to two sentences afterwards
        #     # Prafulla: May include this later
        #     break
        else:
            candidates.append(get_candidate(child, candidate_node, parent_ID, label))
    return candidates


def get_candidate(child, candidate_node, parent_ID, label):
    if label:
        # Training on labels, so either add label or 'NO_EDGE'
        if candidate_node.ID == parent_ID:
            return candidate_node, child, label
        else:
            return candidate_node, child, 'NO_EDGE'
    else:
        # Predicting labels with model, so add None here
        return candidate_node, child, None


def make_training_data(train_file):
    """ Given a file of multiple documents in ConLL-similar format,
    produce a list of training docs, each training doc is
    (1) a list of sentences in that document; and
    (2) a list of (parent_candidate, child_node, edge_label/no_edge) tuples
    in that document;
    and the vocabulary of this training data set.
    """

    data = codecs.open(train_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')
    folder_name = train_file.split('/')[-1].split('.')[0]

    training_data = []

    for doc in doc_list:
        try:
            training_data.append(make_one_doc_training_data(doc, folder_name))
        except ValueError as e:
            print('WARNING: {}, skipping document'.format(e))
            pass

    return training_data


def make_one_doc_test_data(doc, folder_name):
    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    doc_name = doc[0].split('doc id="')[1].split('"')[0]
    snt_list, edge_list = create_snt_edge_lists(doc)

    # create node_dict
    node_dict, node_type = create_node_dict(snt_list, edge_list)
    dp_tags = get_discourse_tags(folder_name, doc_name)

    # create test instance list
    test_instance_list = []
    root_node = get_root_node()
    for c_key in node_dict:
        c_node = node_dict[c_key]
        if c_node.full_label == 'Timex':
            test_instance_list.append(choose_candidates(node_dict, root_node, c_node, None, None, 'Timex'))
        else:
            test_instance_list.append(choose_candidates(node_dict, root_node, c_node, None, None, 'Timex'))
            test_instance_list.append(choose_candidates(node_dict, root_node, c_node, None, None, 'Event'))

    return [snt_list, test_instance_list, dp_tags, edge_list]


def make_test_data(test_file):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')
    folder_name = test_file.split('/')[-1].split('.')[0]

    test_data = []

    for doc in doc_list:
        test_data.append(make_one_doc_test_data(doc, folder_name))

    return test_data
