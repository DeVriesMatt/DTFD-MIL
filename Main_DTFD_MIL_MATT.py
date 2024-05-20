import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import pickle5
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


parser = argparse.ArgumentParser(description='abc')
testMask_dir = '/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/testing/lesion_annotations'  ## Point to the Camelyon test set mask location

parser.add_argument('--task', default='evaluate', type=str)
parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)  ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='./features_256_train.pickle', type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default='./features_256_val.pickle', type=str)  ## Validation Set
parser.add_argument('--mDATA_dir_test0', default='/home/mvries/Documents/GitHub/DTFD-MIL/features_256_test.pickle', type=str)  ## Test Set
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)  ## MaxMinS, MaxS, AFS
parser.add_argument('--saved_model_path',
                    default='/home/mvries/Documents/GitHub/DTFD-MIL/debug_log/best_model.pth', type=str)  ## MaxMinS, MaxS, AFS

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)


def extract_feats_for_heatmap(data, save_path):
    params = parser.parse_args()

    in_chn = 512

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(
        params.device)

    print(save_path)
    checkpoint = torch.load(save_path, map_location='cuda')
    print("model loaded")

    classifier.load_state_dict(checkpoint['classifier'])

    attention.load_state_dict(checkpoint['attention'])
    dimReduction.load_state_dict(checkpoint['dim_reduction'])
    attCls.load_state_dict(checkpoint['att_classifier'])

    # with open(params.mDATA_dir_test0, 'rb') as f:
    #     mDATA_test = pickle5.load(f)
    # # import pandas as pd
    # # mDATA_test = pd.read_pickle(params.mDATA_dir_test0)


    # TODO: Change this whenchanging datasets.
    SlideNames_test, FeatList_test, Label_test = data

    logits, Y_prob, Y_hat, A, _, indexes = infer_single_slide(
        classifier=classifier, dimReduction=dimReduction,
        attention=attention,
        UClassifier=attCls,
        mDATA_list=([SlideNames_test], [FeatList_test], [Label_test]),
        # TODO: Changed to only see first slide for extractning heatmaps
        criterion=None, params=params, numGroup=params.numGroup_test,
        total_instance=params.total_instance_test,
        distill=params.distill_type)

    return logits, Y_prob, Y_hat, A, indexes


def infer_single_slide(mDATA_list,
                           classifier,
                           dimReduction,
                           attention,
                           UClassifier,
                           criterion=None,
                           params=None,
                       numGroup=3,
                           total_instance=3, distill='MaxMinS'):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list
    instance_per_group = total_instance // numGroup


    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = 1
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))
        A = []

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            batch_feat = [FeatLists[sst].to(params.device) for sst in tidx_slide]

            for tidx, tfeat in enumerate(batch_feat):
                midFeat = dimReduction(tfeat)

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
                A.append(AA)
                allSlide_pred_softmax = []
                all_feat_indexes = []
                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []
                    all_patch_pred_softmax = []
                    all_patch_pred_logits = []
                    for tindex in index_chunk_list:
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)

                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                        all_patch_pred_softmax.append(patch_pred_softmax)
                        all_patch_pred_logits.append(patch_pred_logits)
                        print(tindex)
                        all_feat_indexes.append(tindex)

                    all_patch_pred_softmax = torch.cat(all_patch_pred_softmax, dim=0)
                    all_patch_pred_logits = torch.cat(all_patch_pred_logits, dim=0)



                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)

                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                A = torch.cat(A)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gPred_0 = torch.cat([gPred_0], dim=0)
                all_indexes = torch.cat([torch.tensor(x) for x in all_feat_indexes], dim=0)



    logits = all_patch_pred_logits
    indexes = all_indexes
    Y_prob = gPred_1
    Y_hat = gPred_1.argmax(axis=1)


    A = A
    # print(f"A == : {A}")
    # A = AA
    print("logits: ", logits)
    print("Indexes:" , indexes)

    return logits, Y_prob, Y_hat, A, AA, indexes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA):
    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(mDATA):
    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA_v2(mDATA):
    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')

        Label.append(label)
        featGroup = mDATA[slide_name]

        # featGroup = []
        # for tpatch in patch_data_list:
        #     tfeat = torch.from_numpy(tpatch['feature'])
        #     featGroup.append(tfeat.unsqueeze(0))

        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA_test_v2(mDATA):
    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        featGroup = mDATA[slide_name]

        # featGroup = []
        # for tpatch in patch_data_list:
        #     tfeat = torch.from_numpy(tpatch['feature'])
        #     featGroup.append(tfeat.unsqueeze(0))

        FeatList.append(featGroup)
    return SlideNames, FeatList, Label


import pandas as pd
def reorganize_mData_sarcoma(mDATA):
    SlideNames = []
    FeatList = []
    Label = []
    labels = pd.read_csv("/mnt/nvme0n1/ICCV/lipo/lipo_sub_data.csv")
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)
        label = labels[labels['slide_id'] == slide_name]['slide_label'].values[0]
        assert (label == 1) or (label == 0)

        Label.append(label)
        featGroup = mDATA[slide_name]
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label
