

"""
Cityscapes suffixes in gtFine 
    _gtFine_color.png
    _gtFine_instanceIds.png
    _gtFine_labelIds.png
    _gtFine_labelTrainIds.png
    _gtFine_polygons.json
"""

import os
import cv2 
import json
import argparse

import numpy as np

from glob import glob 
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to create color label", type=str)
parser.add_argument("--dataset_type", default='Cityscapes', type=str, help="Dataset type",
                    choices=['Cityscapes'])
parser.add_argument("--split", nargs='+', default=['train', 'val', 'test'], type=str,
                    help="Split to be converted")
parser.add_argument("--mode", default='overwrite', choices=['overwrite', 'skip'], type=str,
                    help="Split to be converted")
parser.add_argument("--palette", default=None, type=json.loads,
                    help="Color palette")
    
args = parser.parse_args()

SUPPORTED_DATASETS = ['Cityscapes']
SUPPORTED_SPLITS = ['train', 'val', 'test']

PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255]]


def is_nested_list(input):
    """Check if a list is nested. 
    Args: 
        input (list)

    Returns: 
        is_nested (bool)
    """

    if isinstance(input, list): 
        return any(isinstance(i, list) for i in input)
        
    else:
        return False
        
    

def main():

    source_path = args.source_path
    dataset_type = args.dataset_type
    splits = args.split
    palette = args.palette

    for split in splits:
        assert split in SUPPORTED_SPLITS, \
            f"Dataset split should be one of {SUPPORTED_SPLITS}."
    
    if not palette : 
        palette = PALETTE

    assert is_nested_list(palette), "Color palette should be a nested list."

    img_list = []
    for split in splits: 
        img_list += glob(os.path.join(source_path, 'leftImg8bit', split, '*_leftImg8bit.png' ))


    for img_path in tqdm(img_list, desc='Creating Colormap and ColorOverlap'): 
        gt_path = img_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png') 
        gt_path = gt_path.replace('leftImg8bit/', 'gtFine/') 

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        gt_color = np.zeros_like(img)

        for i in range(len(PALETTE)):
            color = np.array(PALETTE[i])
            img[gt == i] = img[gt == i] * 0.6 + color * 0.4
            gt_color[gt == i] = color

        colorOverlap_path = gt_path.replace('_gtFine_labelIds.png', '_gtFine_colorOverlap.png')
        color_path = gt_path.replace('_gtFine_labelIds.png', '_gtFine_color.png')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(colorOverlap_path, img)

        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(color_path, gt_color)
    

if __name__ == '__main__':
    

    main()

