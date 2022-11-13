from data_preparation import make_training_data, make_test_data
from data_structures import EDGE_LABEL_COMPRESSED
import random
import torch
import numpy as np
import torch.nn as nn
from eval_util import evaluate
from model import Classifier_DP
from transformers import *
from util import get_features, get_dp_features


LEN_EDGES = len(EDGE_LABEL_COMPRESSED)
MAX_SEGMENT_SIZE = 96


def train(data, model, tokenizer, device, batch_size=5):
    model.train()
    optimizer.zero_grad()
    dp_loss = 0
    total_loss = 0
    random.shuffle(data)
    for batch_index in range(0, len(data), batch_size):
        for document in data[batch_index: min(batch_index+batch_size, len(data))]:
            for sentences, dp_labels in get_dp_features(document, tokenizer, MAX_SEGMENT_SIZE, device):
                dp_labels = torch.LongTensor(dp_labels)
                dp_labels = dp_labels.to(device)
                dp_output = model.forward(sentences, True)
                loss = criterion(dp_output, dp_labels)
                loss.backward()
                dp_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        for document in data[batch_index: min(batch_index+batch_size, len(data))]:
            batches = get_features(document, tokenizer, device, MAX_SEGMENT_SIZE)
            for mini_batch_event_pairs, mini_batch_labels in batches:
                mini_batch_labels = torch.LongTensor(mini_batch_labels)
                mini_batch_labels = mini_batch_labels.to(device)
                mini_batch_output = model.forward(mini_batch_event_pairs)
                mini_batch_output = mini_batch_output.view(mini_batch_labels.size(0), -1)
                loss = criterion(mini_batch_output, mini_batch_labels)
                loss.backward()
                total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    print("Training Loss: ", total_loss, "DP Loss: ", dp_loss, "learning rate:", scheduler.get_lr()[0])


if __name__ == '__main__':

    devices = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("GPU Counts {0}".format(torch.cuda.device_count()))
    TRANSFORMER = "roberta-base"
    seed_val = 42
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    gold_training_data = make_training_data("../tdg_data/train.txt")
    gold_development_data = make_test_data("../tdg_data/dev.txt")

    model = Classifier_DP(out_dim=1, transformer=TRANSFORMER)
    tokenizer = RobertaTokenizer.from_pretrained(TRANSFORMER)
    model = nn.DataParallel(model)
    model = model.to(devices)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(params, lr=8e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=6000)
    criterion = nn.CrossEntropyLoss()

    best_dev_loss = 0  #evaluate(gold_development_data, model, tokenizer, devices)
    for epoch in range(15):
        print("---------------------------Started Training Epoch = {0}--------------------------".format(epoch+1))
        train(data=gold_training_data, model=model, tokenizer=tokenizer, device=devices, batch_size=1)
        dev_loss = evaluate(gold_development_data, model, tokenizer, devices)
        if dev_loss > best_dev_loss:
            best_dev_loss = dev_loss
            print("Best Development Loss: {0}, Started saving model in epoch # {1}".format(best_dev_loss, epoch+1))
            torch.save(model.state_dict(), 'TDG_MODEL_DP_8e5.pt')
