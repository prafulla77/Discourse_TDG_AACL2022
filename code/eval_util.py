from data_structures import EDGE_LABEL_COMPRESSED
import torch
from util import get_features, get_label_features_test
import pickle


MAX_EVAL_SEGMENT_SIZE = 320


def get_labeled_tuple_set(tups):
    # Omit child_label
    tup_set = set([(tup[0], tup[2], tup[3]) for tup in tups])
    return tup_set


def get_unlabeled_tuple_set(tups):
    # Omit child label and link label
    tup_set = set([(tup[0], tup[2]) for tup in tups])
    return tup_set


def is_cycle(edge, children_dict):
    child = edge[0]
    parent = edge[2]
    if child not in children_dict:
        return False
    if parent not in children_dict[child]:
        return False
    return True


def update_parent_children(children_dict, parent_dict, predicted_edge):
    child = predicted_edge[0]
    parent = predicted_edge[2]
    parent_dict[child] = parent  # It will see child for the first time

    parent_dict[child] = parent
    children_dict[parent] = children_dict.get(parent, {})
    children_dict[parent].update({child: None})
    children_dict[parent].update(children_dict.get(child, {}))

    while parent in parent_dict:
        parent = parent_dict[parent]
        children_dict[parent] = children_dict.get(parent, {})
        children_dict[parent].update({child: None})
        children_dict[parent].update(children_dict.get(child, {}))


def decode(full_label_ranks, full_labels):
    all_edges = []
    parent_dict = {}
    children_dict = {}
    for batch_index, label_ranks in enumerate(full_label_ranks):
        labels = full_labels[batch_index]
        for index, ranks in enumerate(label_ranks):
            j = 0
            while j < len(ranks):
                if not is_cycle(labels[index][ranks[j]], children_dict):
                    break
                j += 1
            assert j < len(ranks), "Cycle Detected!"
            all_edges.append(labels[index][ranks[j]])
            update_parent_children(children_dict, parent_dict, labels[index][ranks[j]])
    return all_edges


def calculate_tp_fp_fr(parsed_tup_set, gold_tup_set):
    true_positive = len(gold_tup_set.intersection(parsed_tup_set))
    false_positive = len(parsed_tup_set.difference(gold_tup_set))
    false_negative = len(gold_tup_set.difference(parsed_tup_set))
    return true_positive, false_positive, false_negative


def score_f1(counts):
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])
    p = true_p / (true_p + false_p)
    r = true_p / (true_p + false_n)
    f = 2 * p * r / (p + r) if p + r != 0 else 0
    print('micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    return f


def evaluate(data, model, tokenizer, device, fname="test.pkl"):
    model.eval()
    counts = []
    answers = []
    with torch.no_grad():
        for document in data:
            gold_labels = document[3]
            full_label_ranks = []
            full_labels = []
            batches = get_features(document, tokenizer, device, MAX_EVAL_SEGMENT_SIZE, is_train=False)
            for mini_batch_event_pairs, mini_batch_labels in batches:
                mini_batch_output = model.forward(mini_batch_event_pairs)
                # print(mini_batch_output.size(), len(mini_batch_labels))
                mini_batch_output = mini_batch_output.view(len(mini_batch_labels), -1)
                label_rank = torch.argsort(mini_batch_output, dim=1, descending=True)
                full_labels.append(mini_batch_labels)
                full_label_ranks.append(label_rank.tolist())

            predicted_edges = decode(full_label_ranks, full_labels)
            counts.append(
                calculate_tp_fp_fr(get_unlabeled_tuple_set(predicted_edges), get_unlabeled_tuple_set(gold_labels)))
            assert len(predicted_edges), len(gold_labels)
            answers.append((predicted_edges, gold_labels))

    f = score_f1(counts)
    with open(fname, 'wb') as fp: pickle.dump(answers, fp)
    # print('micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))

    return f


def get_unlabeled_candidate_pairs(data, model, tokenizer, device):
    model.eval()
    answers = []
    with torch.no_grad():
        for document in data:
            full_label_ranks = []
            full_labels = []
            batches = get_features(document, tokenizer, device, MAX_EVAL_SEGMENT_SIZE, is_train=False)
            for mini_batch_event_pairs, mini_batch_labels in batches:
                mini_batch_output = model.forward(mini_batch_event_pairs)
                mini_batch_output = mini_batch_output.view(len(mini_batch_labels), -1)
                label_rank = torch.argsort(mini_batch_output, dim=1, descending=True)
                full_labels.append(mini_batch_labels)
                full_label_ranks.append(label_rank.tolist())

            predicted_edges = decode(full_label_ranks, full_labels)
            answers.append((document, predicted_edges))
    return answers


def evaluate_cls(data, model, tokenizer, device, fname="test.pkl"):
    model.eval()
    answers = []
    counts = []
    with torch.no_grad():
        for document, unlabeled_edges in data:
            gold_labels = document[2]
            predicted_edges = []
            batches = get_label_features_test(document, unlabeled_edges, tokenizer, device, MAX_EVAL_SEGMENT_SIZE)
            for mini_batch_event_pairs, mini_batch_labels in batches:
                mini_batch_output = model.forward(mini_batch_event_pairs)
                # print (mini_batch_output.size())
                _, predict = torch.max(mini_batch_output, 1)
                # print(predict, "-----", mini_batch_labels)
                for id, edge in enumerate(mini_batch_labels):
                    predicted_edges.append(edge+[EDGE_LABEL_COMPRESSED[predict[id]]])
            # print(predicted_edges)
            answers.append((predicted_edges, gold_labels))
            counts.append(
                calculate_tp_fp_fr(get_labeled_tuple_set(predicted_edges), get_labeled_tuple_set(gold_labels)))
            # assert len(predicted_edges)==len(gold_labels)

    f = score_f1(counts)
    # with open(fname, 'wb') as fp: pickle.dump(answers, fp)

    return f
