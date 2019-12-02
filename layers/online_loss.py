# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F


class OnlineLoss(object):
    def __init__(self, margin=0):
        self.margin = margin

    def __call__(self, feats, train_feats, picked_index):
        loss = 0
        counter = 0
        for index in picked_index:
            i, pos, neg = index
            if pos is None and neg is not None:  # contrastive loss
                train_feat = feats[i]
                neg_feat = train_feats[neg]
                dist = ((train_feat - neg_feat) ** 2).sum().sqrt()
                loss += F.relu(-dist + self.margin)
                counter += 1
            elif pos is not None and neg is not None:  # triplet loss
                train_feat = feats[i]
                pos_feat = train_feats[pos]
                neg_feat = train_feats[neg]
                pos_dist = ((train_feat - pos_feat) ** 2).sum().sqrt()
                neg_dist = ((train_feat - neg_feat) ** 2).sum().sqrt()
                loss += F.relu(pos_dist - neg_dist + self.margin)
                counter += 1
            else:
                continue
        if counter == 0:
            return 0
        else:
            return loss / counter
