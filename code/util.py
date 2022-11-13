import torch
from data_structures import EDGE_LABEL_COMPRESSED, DP_MAP


LEN_EDGES = len(EDGE_LABEL_COMPRESSED)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def convert_examples_to_features(examples, tokenizer, seq_length=75):
    # `<s> d e f </s> </s> 1 2 3 </s>`
    features = []
    for example in examples:
        if len(example) == 2:  # Sentence Pair
            input_ids = tokenizer.convert_tokens_to_ids(['<s>'] + example[0] + ['</s>', '</s>'] + example[1] + ['</s>'])
        else:
            input_ids = tokenizer.convert_tokens_to_ids(['<s>'] + example[0] + ['</s>'])
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(1)  # RoBERTa Pad ID is 1
            input_mask.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask))

    return features


def get_contextualized_representation(event_pairs, tokenizer, seq_length, device):
    features = convert_examples_to_features(examples=event_pairs, tokenizer=tokenizer, seq_length=seq_length)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    return {'input_ids': input_ids, 'input_mask': input_mask}


def get_labeled_index(label_id):
    return label_id[0]*LEN_EDGES + EDGE_LABEL_COMPRESSED.index(label_id[1])


def get_unlabeled_index(label_id):
    return label_id[0]


def process_pair(parent, child, sentences, tokenizer):
    s1 = sentences[parent.snt_index_in_doc]
    s2 = sentences[child.snt_index_in_doc]
    c11 = tokenizer.tokenize(' '.join(s1[: parent.start_word_index_in_snt]))
    c12 = tokenizer.tokenize(' '.join(['@'] + s1[parent.start_word_index_in_snt: parent.end_word_index_in_snt+1] + ['@']))
    # c12 = tokenizer.tokenize(' '.join(s1[parent.start_word_index_in_snt: parent.end_word_index_in_snt+1]))
    c13 = tokenizer.tokenize(' '.join(s1[parent.end_word_index_in_snt+1:]))

    c21 = tokenizer.tokenize(' '.join(s2[: child.start_word_index_in_snt]))
    c22 = tokenizer.tokenize(' '.join(['$'] + s2[child.start_word_index_in_snt: child.end_word_index_in_snt+1] + ['$']))
    # c22 = tokenizer.tokenize(' '.join(s2[child.start_word_index_in_snt: child.end_word_index_in_snt+1]))
    c23 = tokenizer.tokenize(' '.join(s2[child.end_word_index_in_snt+1:]))

    c1 = c11[max(0, len(c11)-50):] + c12 + c13[:min(len(c13), 51)]  #+ tokenizer.tokenize(dp_tags[parent.snt_index_in_doc])
    c2 = c21[max(0, len(c21)-50):] + c22 + c23[:min(len(c23), 51)]  #+ tokenizer.tokenize(dp_tags[child.snt_index_in_doc])

    if parent.snt_index_in_doc <= child.snt_index_in_doc:
        return [c1, c2]
    else:
        return [c2, c1]



def get_features(document, tokenizer, device, MAX_SEGMENT_SIZE, is_train=True):
    sentences = document[0]
    # dp_tags = document[2]
    batches = []
    event_pairs = []
    labels = []
    previous_length = 0
    for relation in document[1]:
        # Make a new batch, number of possible event-timex pairs changed
        if len(relation) != previous_length or len(event_pairs) > MAX_SEGMENT_SIZE:
            previous_length = len(relation)
            if len(event_pairs) > 0:
                seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)  # ( 1 <s> and 3 </s>)
                event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
                batches.append((event_pairs, labels))
                event_pairs, labels = [], []

        # Process data, tokenize, etc.
        # Child marked as @ child @ and Parent marked as $ parent $
        # Ensure sentence order remains consistent with the document
        if is_train:
            label_id = None
        else:
            labels.append([])
        for index, (parent, child, label) in enumerate(relation):
            event_pairs.append(process_pair(parent, child, sentences, tokenizer))

            if is_train:
                if label != 'NO_EDGE':
                    if not label_id:
                        label_id = (index, label)
                    else:
                        print("Multiple labels", relation)
            else:
                # For labeled edges
                # for edge in EDGE_LABEL_COMPRESSED:
                #     labels[-1].append([child.ID, child.full_label, parent.ID, edge])
                # For unlabeled
                labels[-1].append([child.ID, child.full_label, parent.ID, None])

        if is_train:
            # labels.append(get_labeled_index(label_id))
            labels.append(get_unlabeled_index(label_id))

    # Include the last batch
    if len(event_pairs) > 0:
        seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)
        event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
        batches.append((event_pairs, labels))
    return batches


def get_label_features(document, tokenizer, device, MAX_SEGMENT_SIZE, is_train=True):
    sentences = document[0]
    # dp_tags = document[2]
    batches = []
    event_pairs = []
    labels = []

    for relation in document[1]:
        # Make a new batch, number of possible event-timex pairs changed
        if len(event_pairs) > MAX_SEGMENT_SIZE:
            previous_length = len(relation)
            if len(event_pairs) > 0:
                seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)  # ( 1 <s> and 3 </s>)
                event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
                batches.append((event_pairs, labels))
                event_pairs, labels = [], []

        # Process data, tokenize, etc.
        # Child marked as @ child @ and Parent marked as $ parent $
        # Ensure sentence order remains consistent with the document
        for index, (parent, child, label) in enumerate(relation):
            if label != 'NO_EDGE':
                event_pairs.append(process_pair(parent, child, sentences, tokenizer))
                if is_train:
                    labels.append(EDGE_LABEL_COMPRESSED.index(label))
                else:
                    labels.append([child.ID, child.full_label, parent.ID, label])

    # Include the last batch
    if len(event_pairs) > 0:
        seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)
        event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
        batches.append((event_pairs, labels))
    return batches


def get_label_features_test(document, predicted_edges, tokenizer, device, MAX_SEGMENT_SIZE):
    sentences = document[0]
    batches = []
    event_pairs, labels = [], []
    total_included = 0

    predicted_edges_map = {}
    for edge in predicted_edges:
        if (edge[0], edge[2]) not in predicted_edges_map:
            predicted_edges_map[(edge[0], edge[2])] = 1
        else:
            predicted_edges_map[(edge[0], edge[2])] += 1

    for relation in document[1]:
        # Make a new batch, number of possible event-timex pairs changed
        if len(event_pairs) > MAX_SEGMENT_SIZE:
            seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)  # ( 1 <s> and 3 </s>)
            event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
            batches.append((event_pairs, labels))
            event_pairs, labels = [], []

        # Process data, tokenize, etc.
        # Child marked as @ child @ and Parent marked as $ parent $
        # Ensure sentence order remains consistent with the document
        for index, (parent, child, label) in enumerate(relation):
            if (child.ID, parent.ID) in predicted_edges_map and predicted_edges_map[(child.ID, parent.ID)] > 0:
            # if label != 'NO_EDGE':
            #     print(child.ID, child.full_label, parent.ID)
                labels.append([child.ID, child.full_label, parent.ID])
                event_pairs.append(process_pair(parent, child, sentences, tokenizer))
                total_included += 1
                predicted_edges_map[(child.ID, parent.ID)] -= 1

    # Include the last batch
    if len(event_pairs) > 0:
        seq_length = max([len(elem[0] + elem[1]) for elem in event_pairs]) + 4  # ( 1 <s> and 5 </s>)
        event_pairs = get_contextualized_representation(event_pairs, tokenizer, seq_length, device)
        batches.append((event_pairs, labels))
    # print(total_included, len(predicted_edges))
    assert total_included==len(predicted_edges)
    return batches


def get_dp_features(document, tokenizer, MAX_SEGMENT_SIZE, device):
    sentences = []
    dp_tags = document[2]
    output_labels = []
    batches = []
    sentence_pairs = []
    for sentence in document[0]:
        sentences.append(tokenizer.tokenize(' '.join(sentence)))
    for temp_id, temp_sent in enumerate(sentences):
        sentence_pairs.append([temp_sent[:min(len(temp_sent), 198)]])
        output_labels.append(dp_tags[temp_id])
        if len(sentence_pairs) > MAX_SEGMENT_SIZE:
            sequence_len = max([len(elem[0]) for elem in sentence_pairs]) + 2  # for <s> and </s>
            sentence_tokens = get_contextualized_representation(sentence_pairs, tokenizer, sequence_len, device)
            batches.append((sentence_tokens, output_labels))
            sentence_pairs = []
            output_labels = []
    if len(sentence_pairs) > 0:
        sequence_len = max([len(elem[0]) for elem in sentence_pairs]) + 2
        sentence_tokens = get_contextualized_representation(sentence_pairs, tokenizer, sequence_len, device)
        batches.append((sentence_tokens, output_labels))
    # print(output_labels)
    return batches


def get_dp_features_pair(document, tokenizer, MAX_SEGMENT_SIZE, device):
    sentences = []
    dp_tags = document[2]
    output_labels = []
    batches = []
    sentence_pairs = []
    for sentence in document[0]:
        sentences.append(tokenizer.tokenize(' '.join(sentence)))
    for temp_id, temp_sent in enumerate(sentences):
        for prev_id, prev_sent in enumerate(sentences[:temp_id]):
            if dp_tags[temp_id] != 0 and dp_tags[prev_id] != 0:
                sentence_pairs.append([prev_sent[:min(len(prev_sent), 98)], temp_sent[:min(len(temp_sent), 98)]])
                output_labels.append(DP_MAP.get((dp_tags[prev_id], dp_tags[temp_id]), 0))

                if len(sentence_pairs) > MAX_SEGMENT_SIZE:
                    sequence_len = max([len(elem[0] + elem[1]) for elem in sentence_pairs]) + 4  # for <s> and 3 </s>
                    sentence_tokens = get_contextualized_representation(sentence_pairs, tokenizer, sequence_len, device)
                    batches.append((sentence_tokens, output_labels))
                    sentence_pairs = []
                    output_labels = []
    if len(sentence_pairs) > 0:
        sequence_len = max([len(elem[0] + elem[1]) for elem in sentence_pairs]) + 4  # for <s> and 3 </s>
        sentence_tokens = get_contextualized_representation(sentence_pairs, tokenizer, sequence_len, device)
        batches.append((sentence_tokens, output_labels))
    # print(output_labels)
    return batches

