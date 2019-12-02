# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.engine import Events


class DistributionMetric(Metric):
    def __init__(self):
        super(DistributionMetric, self).__init__()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        feats = F.normalize(feats, p=2, dim=1)  # normalized feature
        # query
        qf = feats[:]
        q_pids = np.asarray(self.pids)

        # gallery
        gf = feats[:]
        g_pids = np.asarray(self.pids)

        m, n = qf.shape[0], gf.shape[0]
        similarity = torch.mm(qf, gf.t())
        similarity = similarity.cpu().numpy()

        pos_similarity = []
        neg_similarity = []
        for i, q_index in enumerate(q_pids):
            pos_idx = np.where(g_pids == q_index)[0]
            same_idx = np.where(pos_idx == i)[0]
            pos_idx = np.delete(pos_idx, same_idx)
            neg_idx = np.where(g_pids != q_index)[0]
            pos_similarity.extend(similarity[i, pos_idx])
            neg_similarity.extend(similarity[i, neg_idx])
        pos_similarity = np.asarray(pos_similarity)
        neg_similarity = np.asarray(neg_similarity)
        t1 = pos_similarity.min()
        t2 = neg_similarity.max()

        return t1, t2, pos_similarity, neg_similarity


class Sample(Metric):
    def __init__(self, output_transform):
        super(Sample, self).__init__(output_transform=output_transform)

    def reset(self):
        self.pos_pick = 0
        self.pos_total = 0
        self.neg_pick = 0
        self.neg_total = 0

    def update(self, output):
        pos_pick, pos_total, neg_pick, neg_total = output
        self.pos_pick += pos_pick
        self.pos_total += pos_total
        self.neg_pick += neg_pick
        self.neg_total += neg_total

    def compute(self):
        return self.pos_pick, self.pos_total, self.neg_pick, self.neg_total

    def attach(self, engine, name):
        # restart average every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)
