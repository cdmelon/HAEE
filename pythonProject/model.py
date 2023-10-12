import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import time
from transformers import BertPreTrainedModel, BertModel
from data_utils import get_example_rel, get_event_rel

import os
import json
import codecs


def json2dicts(jsonFile):
    data = []
    with codecs.open(jsonFile, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    return data


logger = logging.getLogger(__name__)

device = torch.device("cuda")

num = 0

def set_device(de):
    global device
    device = de


class NewModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.prototypes = nn.Embedding(config.num_labels, config.hidden_size).to(device)
        self.proto_size = config.num_labels

        # 极坐标

        assert config.hidden_size % 2 == 0

        self.hidden_dim = int(config.hidden_size / 2)

        self.gamma = nn.Parameter(
            torch.Tensor([7]).to(device),
            requires_grad=False
        )
        self.epsilon = 8.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]).to(device),
            requires_grad=False
        )

        self.prototypes = nn.Embedding(self.proto_size, self.hidden_dim * 2).to(device)

        self.relation_embedding = nn.Parameter(torch.zeros(self.hidden_dim * 3).to(device))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[self.hidden_dim:2 * self.hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[2 * self.hidden_dim:3 * self.hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]).to(device),
                                         requires_grad=False)
        self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]).to(device))

        self.pi = 3.14159262358979323846
        self.ratio_proto_emb = 0.4
        self.margin = 0.08
        self.r_gamma = 8
        self.wandb = None
        self.emb = 100
        self.hid = config.hidden_size
        self.re_classifier = None

        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 3).to(device))

    def set_wandb(self, wandb):
        self.wandb = wandb

    def set_config(self, config, flag=False):
        if flag:
            self.ratio_proto_emb = config.ratio_proto_emb
            self.margin = config.margin
            self.r_gamma = config.r_gamma
            self.emb = config.emb

        # self.emb = config.emb
        self.re_classifier = nn.Linear(self.hid, self.emb).to(device)
        self.prototypes = nn.Embedding(self.proto_size, self.emb).to(device)

        # tmp=(self.r_gamma + self.epsilon) / self.emb
        tmp = self.r_gamma / (self.emb/2)
        nn.init.uniform_(
            tensor=self.prototypes.weight,
            a=-tmp,
            b=tmp
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([tmp]).to(device),
            requires_grad=False
        )
        self.phase_weight = nn.Parameter(torch.tensor([[0.5 * tmp]]).to(device))

    def __dist__(self, x, y, dim):
        dist = torch.pow(x - y, 2).sum(dim)
        # dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1e-8), dist)
        return dist

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def proto_emb(self, id, is_ca, is_head):
        # proto_embedding = self.prototypes
        proto_embedding = self.prototypes(torch.tensor(range(0, self.proto_size)).to(device))
        rel_event_ids = get_event_rel()
        h1 = proto_embedding[id]
        t = 0
        h2 = torch.zeros(self.emb).to(device)
        if is_ca:
            if is_head:
                for it in rel_event_ids[id][1]:
                    h2 += proto_embedding[it[0]]
                    t += 1
            else:
                for it in rel_event_ids[id][0]:
                    h2 += proto_embedding[it[1]]
                    t += 1
        else:
            if is_head:
                for it in rel_event_ids[id][3]:
                    h2 += proto_embedding[it[0]]
                    t += 1
            else:
                for it in rel_event_ids[id][2]:
                    h2 += proto_embedding[it[1]]
                    t += 1
        if t != 0:
            h2 /= t
            h1 = h1 * self.ratio_proto_emb + h2 * (1 - self.ratio_proto_emb)
        return h1

    def func_ca(self, ca_h, ca_t, rate):
        phase_relation, mod_relation, bias_relation = torch.chunk(self.relation_embedding, 3, dim=0)
        loss = torch.tensor(0).to(device)
        if len(ca_h) != 0:
            ca_h = ca_h.unsqueeze(1)
            ca_t = ca_t.unsqueeze(1)

            # phase_ca_h, _ = torch.chunk(ca_h, 2, dim=2)
            # phase_ca_t, _ = torch.chunk(ca_t, 2, dim=2)
            phase_ca_h = ca_h[:,:,:round(self.emb*2/3)]
            phase_ca_t = ca_t[:,:,:round(self.emb*2/3)]

            phase_ca_h = phase_ca_h / (self.embedding_range.item() / self.pi)
            # phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_relation = self.embedding_range.item() / (self.embedding_range.item() / self.pi)
            phase_ca_t = phase_ca_t / (self.embedding_range.item() / self.pi)

            phase_score = phase_ca_h + (phase_relation - phase_ca_t)

            phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight

            loss = F.logsigmoid(self.r_gamma - phase_score).squeeze(dim=1)
            loss = -torch.mean(loss)

        return loss

    def func_su(self, su_h, su_t, rate):
        phase_relation, mod_relation, bias_relation = torch.chunk(self.relation_embedding, 3, dim=0)
        loss = torch.tensor(0).to(device)
        if len(su_h) != 0:
            su_h = su_h.unsqueeze(1)
            su_t = su_t.unsqueeze(1)

            _, mod_su_h = torch.chunk(su_h, 2, dim=2)
            _, mod_su_t = torch.chunk(su_t, 2, dim=2)

            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]

            r_score = mod_su_h * (mod_relation + bias_relation) - mod_su_t * (1 - bias_relation)
            r_score = torch.norm(r_score, dim=2) * self.modulus_weight

            loss = F.logsigmoid(self.r_gamma - r_score).squeeze(dim=1)
            loss = rate * -torch.mean(loss) + (1 - rate) * -torch.mean(torch.topk(loss,self.max_p).values)
        return loss

    def func_su_new(self, su_a, su_p, su_n, rate=0):
        phase_relation, mod_relation, bias_relation = torch.chunk(self.relation_embedding, 3, dim=0)
        loss = torch.tensor(0).to(device)
        if len(su_a) != 0:
            margin = self.margin
            su_a = su_a[:,round(self.emb/3):]
            su_p = su_p[:,round(self.emb/3):]
            su_n = su_n[:,round(self.emb/3):]
            criterion = nn.TripletMarginLoss(margin=margin)
            loss = criterion(su_a, su_p, su_n)
        return loss

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, example_ids=None):
        batch_size = input_ids.size(0)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = []
        for i in range(2):
            seq = outputs[2][-i]
            pooled_output += [torch.mean(seq, dim=1, keepdim=True)]
        pooled_output = torch.sum(torch.cat(pooled_output, dim=1), 1)
        pooled_output = F.relu(pooled_output)
        instance_embedding = self.dropout(pooled_output)

        torch.autograd.set_detect_anomaly(True)

        # proto_embedding = self.prototypes
        proto_embedding = self.prototypes(torch.tensor(range(0, self.proto_size)).to(device))
        instance_embedding = self.re_classifier(instance_embedding)

        logits = -self.__batch_dist__(proto_embedding, instance_embedding)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        # 极坐标
        ca_h = torch.tensor([]).to(device)
        ca_t = torch.tensor([]).to(device)
        su_h = torch.tensor([]).to(device)
        su_t = torch.tensor([]).to(device)

        rel_event_ids = get_event_rel()
        for i in labels:
            id = i.item()
            for it in rel_event_ids[id][0]:
                ca_h = torch.cat((ca_h, proto_embedding[id].unsqueeze(0)), 0)
                ca_t = torch.cat((ca_t, self.proto_emb(it[1], True, False).unsqueeze(0)), 0)
            for it in rel_event_ids[id][1]:
                ca_h = torch.cat((ca_h, self.proto_emb(it[0], True, True).unsqueeze(0)), 0)
                ca_t = torch.cat((ca_t, proto_embedding[id].unsqueeze(0)), 0)
            for it in rel_event_ids[id][2]:
                su_h = torch.cat((su_h, proto_embedding[id].unsqueeze(0)), 0)
                su_t = torch.cat((su_t, self.proto_emb(it[1], False, False).unsqueeze(0)), 0)
            for it in rel_event_ids[id][3]:
                su_h = torch.cat((su_h, proto_embedding[id].unsqueeze(0)), 0)
                su_t = torch.cat((su_t, self.proto_emb(it[0], False, True).unsqueeze(0)), 0)

        loss_p = self.func_ca(ca_h, ca_t, 0.8)
        if len(su_t) > 1:
            loss_r = self.func_su_new(su_t, torch.cat((su_t[:, 1:], su_t[:, :1]), 1), su_h)
        else:
            loss_r = self.func_su_new(su_t, su_t, su_h)

        # loss = self.major  * loss + self.p_rate * loss_p + self.r_rate * loss_r  # +ins_loss_p+ins_loss_r

        loss = loss/(2*self.loss_scale[0].exp())+self.loss_scale[0]/2
        loss += loss_p / (2 * self.loss_scale[1].exp()) + self.loss_scale[1] / 2
        loss += loss_r / (2 * self.loss_scale[2].exp()) + self.loss_scale[2] / 2


        global num
        num += 1

        outputs = (logits,) + outputs[2:]
        outputs = (loss,) + outputs

        return outputs
