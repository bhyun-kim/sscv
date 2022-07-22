
import os
import argparse

from glob import glob 
from tqdm import tqdm

from datetime import datetime

import numpy as np


import cv2 
import slidingwindow as sw

import pathlib


parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to images to be cropped", type=str)
parser.add_argument("store_path", help="file path to store cropped images", type=str)
parser.add_argument("--source_ext", default='jpg', type=str,
                    help="file extension to be cropped")
parser.add_argument("--store_ext", default='jpg', type=str,
                    help="file extension to be cropped")
parser.add_argument("--crop_size", default=256, type=int,
                    help="image size to be cropped")
parser.add_argument("--date_stamp", default=True, type=bool,
                    help="Include the date and time of script operation in the file name")


args = parser.parse_args()



def main():

    source_path = args.source_path
    store_path = args.store_path
    source_ext = args.source_ext
    store_ext = args.store_ext
    crop_size = args.crop_size
    date_stamp = args.date_stamp
    
    img_list = glob(os.path.join(source_path, '*.{}'.format(source_ext)))

    for img_path in tqdm(img_list , desc ='') : 
        img = imread(img_path)
        file_name = os.path.basename(img_path).split('/')[-1]

        windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, crop_size, 0)

        for window in windows:
            subset = img[ window.indices()]

            x_start = window.indices()[0].start
            y_start = window.indices()[1].start
            
            
            file_store_path = os.path.join(store_path, file_name)
            if date_stamp : 
                datetime_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_store_path = file_store_path.replace(f'.{source_ext}', f'_{datetime_stamp}_{x_start}_{y_start}_{crop_size}.{store_ext}')
            else: 
                file_store_path = file_store_path.replace(f'.{source_ext}', f'_{x_start}_{y_start}_{crop_size}.{store_ext}')
            
            imwrite(file_store_path, subset)



def imread(path):
    
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)

    return bgrImage


def imwrite(path, image):
    _, ext = os.path.splitext(path)
    cv2.imencode(ext, image)[1].tofile(path)



if __name__ == '__main__':
    

    main()