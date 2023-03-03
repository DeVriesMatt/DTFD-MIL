import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric


class Model1(nn.Module):
    def __int__(self, params, checkpoint=None):
        super(Model1, self).__init__()
        self.in_chn = 1024
        self.classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
        self.attention = Attention(params.mDim).to(params.device)
        self.dimReduction = DimReduction(self.in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)

        self.instance_per_group = params.total_instance // params.numGroup # 3// 3 in the line 461 of Main_DTFD_MIL in train_attention_preFeature_DTFD
        self.distill = params.distill_type
        self.checkpoint = checkpoint

    def load_from_checkpoint(self):
        if self.checkpoint == None:
            print("No checkpoint given!")

        else:
            try:
                self.classifier.load_state_dict(self.checkpoint['classifier'])
                self.attention.load_state_dict(self.checkpoint['attention'])
                self.dimReduction.load_state_dict(self.checkpoint['dim_reduction'])
            except Exception as e:
                print(f"There was an error: {e}.")

    def forward(self, index_chunk_list, tfeat_tensor, training):
        slide_pseudo_feat = []
        slide_sub_preds = []
        # slide_sub_labels = []

        if not training:
            midFeat = self.dimReduction(tfeat_tensor)
            AA = self.attention(midFeat, isNorm=False).squeeze(0)

        for tindex in index_chunk_list:
            # slide_sub_labels.append(tslideLabel) #Todo: add to training loop
            if training:

                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                index=torch.LongTensor(tindex).to(self.params.device))
                tmidFeat = self.dimReduction(subFeat_tensor)
                tAA = self.attention(tmidFeat).squeeze(0)

            if not training:
                idx_tensor = torch.LongTensor(tindex).to(self.params.device)
                tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)
                tAA = AA.index_select(dim=0, index=idx_tensor)
                tAA = torch.softmax(tAA, dim=0)

            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:self.instance_per_group].long()
            topk_idx_min = sort_idx[-self.instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        return slide_sub_preds, slide_pseudo_feat

