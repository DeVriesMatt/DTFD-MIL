import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import cv2
from scipy.special import softmax
import math
import h5py
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from math import floor
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
import torch.nn as nn
from xml.dom import minidom
from skimage import filters
import multiprocessing as mp
import argparse
import matplotlib

def plot_props(mask):
    properties = measure.regionprops(mask)
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap=plt.cm.gray)

    for props in properties:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    plt.show()


def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.array(slide.read_region((0, 0), level, dims))

    distance = nd.distance_transform_edt(255 - pixelarray[:, :, 0])
    Threshold = 75 / (resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)


    return filled_image


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells


def cam_to_mask(maskDIR,level,patches, scores, thresh):

        slide = openslide.open_slide(maskDIR)
        dims = slide.level_dimensions[level]

        w_s=float(512 / (pow(2, level)))
        h_s=float(512 /(pow(2, level)))
        pixelarray = np.zeros((dims[1],dims[0],4), dtype='float')

        for ind1, patch in enumerate(patches):
            x, y = patch.split('.')[0].split('_')

            x, y = int(x)/pow(2, level), int(y)/pow(2, level)
            if x>dims[0]-w_s or y>dims[1]-h_s:
                 continue
            if scores[ind1] < thresh:
                  continue
            pixelarray[int(y):int(y + h_s), int(x):int(x + w_s)].fill(scores[ind1][0])

        channel = pixelarray[:, :, 0]
        threshold_value = filters.threshold_otsu(channel)
        binary_image = (channel > threshold_value).astype(np.uint8)

        distance_input = 1 - binary_image
        distance = nd.distance_transform_edt(distance_input)
        Threshold = 75 / (0.243 * pow(2, level) * 2)

        binary = distance < Threshold
        return binary


# Dice similarity function
def dice_coefficient(image1, image2):
    """
    Calculate the Dice coefficient for two binary images represented as True and False.

    Parameters:
    - image1: Numpy array representing the first binary image (True and False values).
    - image2: Numpy array representing the second binary image (True and False values).

    Returns:
    - Dice coefficient (a value between 0 and 1).
    """
    # Convert True/False values to 1/0
    image1 = image1.astype(int)
    image2 = image2.astype(int)

    # Calculate intersection (logical AND)
    intersection = np.logical_and(image1, image2)

    # Calculate the sum of pixels in each image
    sum_image1 = np.sum(image1)
    sum_image2 = np.sum(image2)

    # Calculate the Dice coefficient
    dice_coefficient = (2.0 * np.sum(intersection)) / (sum_image1 + sum_image2)

    return dice_coefficient


def calculate_specificity(true_negatives, false_positives):
    """
    Calculate specificity (True Negative Rate) for a binary classification problem.

    Parameters:
    - true_negatives: The number of true negative predictions.
    - false_positives: The number of false positive predictions.

    Returns:
    - Specificity (a float between 0 and 1).
    """
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity


if __name__ == "__main__":

    xml_path='/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/testing/lesion_annotations'

    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--thresh',
                        type=float,
                        default=0.5),
    parser.add_argument('--experiment_name',
                        type=str,
                        default="GTP"),
    parser.add_argument('--output_masks',
                        type=str,
                        default="output_masks",
                        help='threshold value for the probability maps'),
    parser.add_argument('--csv_file', type=str,
                        default='/home/admin_ofourkioti/PycharmProjects/my_models/SAD_MIL/camelyon_csv_files/splits_0.csv',
                        help='slit file'),
    parser.add_argument('--label_file', type=str,
                        default='/home/admin_ofourkioti/PycharmProjects/my_models/SAD_MIL/label_files/camelyon_data.csv')
    parser.add_argument('--path_WSI', type=str,
                        default='/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon_slides/slides/',
                        help='experiment code')
    parser.add_argument('--mask_folder',
                        type=str,
                        default= "/home/admin_ofourkioti/PycharmProjects/baseline_models/TransMIL/camelyon_masks/")
    parser.add_argument('--path_graph',
                        type=str,
                        default="/home/admin_ofourkioti/Documents/data/cam-16/graphs/simclr_files/")
    parser.add_argument('--ckpt_path',
                        type=str, help='ckpt_path',
                        default="/home/admin_ofourkioti/PycharmProjects/paper_results/results/dsmil_feats_clam_sb_s12321/s_0_checkpoint.pt")
    args = parser.parse_args()
    references = pd.read_csv(args.label_file)
    splits_csv = pd.read_csv(args.csv_file)

    def func_val(x):
        value = None
        if isinstance(x, str):
            value = x
        return value

    test_bags = splits_csv.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

    normal_slides = [os.path.splitext(os.path.basename(slide))[0] for slide in test_bags if
                     references["slide_label"].loc[
                         references["slide_id"] == os.path.splitext(os.path.basename(slide))[0]].values.tolist()[
                         0] == 0]

    fold_id = 0
    mask_slides=[]
    for root, dirs, files in os.walk(args.mask_folder):
        for f in files:
            if f.endswith('.tif'):
                mask_slides.append(f)

    EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0

    test_bags = normal_slides + mask_slides

    caseNum = 0


    dice_scores=[]
    specificities = []
    # os.makedirs(os.path.join(args.experimnet_name))
    os.makedirs(os.path.join(args.experiment_name,args.output_masks ), exist_ok=True)
    output_folder= os.path.join(args.experiment_name, args.output_masks)
    for enum, case in enumerate(sorted(test_bags)):
            slide_id = os.path.splitext(os.path.basename(case))[0]

            print('Evaluating Performance on image:', slide_id)

            sys.stdout.flush()


            p = torch.load('/home/admin_ofourkioti/Documents/data/cam-16/graphcam/{}_prob.pt'.format(
                slide_id)).cpu().detach().numpy()[0]

            ori = openslide.OpenSlide(os.path.join(args.path_WSI, '{}.tif').format(slide_id))
            patch_info = open(os.path.join(args.path_graph, slide_id, 'c_idx.txt'), 'r')

            patches = []
            xmax, ymax = 0, 0
            for patch in patch_info:
                x, y = patch.strip('\n').split('\t')
                if xmax < int(x): xmax = int(x)
                if ymax < int(y): ymax = int(y)

                patches.append('{}_{}.jpeg'.format(x, y))

            print('visulize GraphCAM')
            assign_matrix = torch.load(
                '/home/admin_ofourkioti/Documents/data/cam-16/graphcam/{}_s_matrix_ori.pt'.format(slide_id))
            m = nn.Softmax(dim=1)
            assign_matrix = m(assign_matrix)

            # Thresholding for better visualization
            p = np.clip(p, 0.4, 1)

            cam_matrix_0 = torch.load(
                '/home/admin_ofourkioti/Documents/data/cam-16/graphcam/{}_cam_0.pt'.format(slide_id))
            cam_matrix_0 = torch.mm(assign_matrix, cam_matrix_0.transpose(1, 0))
            cam_matrix_0 = cam_matrix_0.cpu()

            cam_matrix_1 = torch.load(
                '/home/admin_ofourkioti/Documents/data/cam-16/graphcam/{}_cam_1.pt'.format(slide_id))
            cam_matrix_1 = torch.mm(assign_matrix, cam_matrix_1.transpose(1, 0))
            cam_matrix_1 = cam_matrix_1.cpu()

            # Normalize the graphcam
            cam_matrix_0 = (cam_matrix_0 - cam_matrix_0.min()) / (cam_matrix_0.max() - cam_matrix_0.min())
            cam_matrix_0 = cam_matrix_0.detach().numpy()
            cam_matrix_0 = p[0] * cam_matrix_0
            cam_matrix_0 = np.clip(cam_matrix_0, 0, 1)
            cam_matrix_1 = (cam_matrix_1 - cam_matrix_1.min()) / (cam_matrix_1.max() - cam_matrix_1.min())
            cam_matrix_1 = cam_matrix_1.detach().numpy()
            cam_matrix_1 = p[1] * cam_matrix_1
            cam_matrix_1 = np.clip(cam_matrix_1, 0, 1)
            is_tumor = references["slide_label"].loc[references["slide_id"] == slide_id].values.tolist()[0]
            if (is_tumor):
                    #maskDIR = os.path.join(args.mask_folder, (slide_id + '_mask.jpg'))
                    maskDIR = os.path.join(args.mask_folder, (slide_id + '.tif'))
                    evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                    mask = cam_to_mask(os.path.join(args.path_WSI, '{}.tif').format(slide_id),
                                                               EVALUATION_MASK_LEVEL, patches, cam_matrix_1, args.thresh)

                    plt.imshow(mask, cmap='gray')
                    plt.axis('off')

                    dice_score = dice_coefficient(mask, evaluation_mask)  # 255 in my case, can be 1

                    plt.savefig(os.path.join(output_folder, slide_id + '.jpeg'),
                                                        bbox_inches='tight', pad_inches=0)
                    plt.close()
                    dice_scores.append(dice_score)

            else:
                    fp = np.count_nonzero(cam_matrix_0 > args.thresh)
                    specificity=calculate_specificity(cam_matrix_0.shape[0], fp)
                    specificities.append(specificity)

            caseNum += 1
    print ('Dice score {}'.format( np.mean(dice_scores)))
    print ('Specificity {}'.format(np.mean(specificities)))
