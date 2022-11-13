import torch.nn as nn
from transformers import *
import torch


class Classifier_DP(nn.Module):

    def __init__(self, out_dim, transformer='roberta-base', dp_dim=10):
        super(Classifier_DP, self).__init__()
        self.encoder_model = RobertaModel.from_pretrained(transformer)
        self.predict = nn.Linear(self.encoder_model.config.hidden_size, out_dim)
        self.DP_predict = nn.Linear(self.encoder_model.config.hidden_size, dp_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.predict.state_dict()['weight'])
        self.predict.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.DP_predict.state_dict()['weight'])
        self.DP_predict.bias.data.fill_(0.)
        print("Model Initialized")

    def get_context(self, context):
        cls_embedding = self.encoder_model(context['input_ids'], attention_mask=context['input_mask']).pooler_output
        return cls_embedding

    def forward(self, event_pairs, dp=False):
        if dp:
            return self.DP_predict(self.get_context(event_pairs))
        return self.predict(self.get_context(event_pairs))


class Classifier(nn.Module):

    def __init__(self, out_dim, transformer='roberta-base'):
        super(Classifier, self).__init__()
        self.encoder_model = RobertaModel.from_pretrained(transformer)
        self.predict = nn.Linear(self.encoder_model.config.hidden_size, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.predict.state_dict()['weight'])
        self.predict.bias.data.fill_(0.)
        print("Model Initialized")

    def forward(self, event_pairs):
        cls_embedding = self.encoder_model(event_pairs['input_ids'], attention_mask=event_pairs['input_mask']).pooler_output
        return self.predict(cls_embedding)
