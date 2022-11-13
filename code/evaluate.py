from data_preparation import make_test_data
from data_structures import EDGE_LABEL_COMPRESSED
import random
import torch
import numpy as np
import torch.nn as nn
from model import Classifier_DP
from transformers import *
from eval_util import evaluate


LEN_EDGES = len(EDGE_LABEL_COMPRESSED)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("GPU Counts {0}".format(torch.cuda.device_count()))
    TRANSFORMER = "roberta-base"
    seed_val = 42
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    gold_test_data = make_test_data("../tdg_data/test.txt")
    gold_development_data = make_test_data("../tdg_data/dev.txt")

    tokenizer = RobertaTokenizer.from_pretrained(TRANSFORMER)
    model = Classifier_DP(out_dim=1, transformer=TRANSFORMER)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load('TDG_MODEL_DP_2e5.pt'))
    print("Model Loaded")

    _ = evaluate(gold_development_data, model, tokenizer, device, "dev.pkl")
    _ = evaluate(gold_test_data, model, tokenizer, device)



