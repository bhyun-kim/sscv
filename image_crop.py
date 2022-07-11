
import os
import argparse

from glob import glob 
from tqdm import tqdm

import cv2 
import slidingwindow as sw


parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to images to be cropped", type=str)
parser.add_argument("store_path", help="file path to store cropped images", type=str)
parser.add_argument("--source_ext", default='jpg', type=str,
                    help="file extension to be cropped")
parser.add_argument("--store_ext", default='jpg', type=str,
                    help="file extension to be cropped")
parser.add_argument("--crop_size", default=256, type=int,
                    help="image size to be cropped")


args = parser.parse_args()



def main():

    source_path = args.source_path
    store_path = args.store_path
    source_ext = args.source_ext
    store_ext = args.store_ext
    crop_size = args.crop_size
    
    img_list = glob(os.path.join(source_path, '*.{}'.format(source_ext)))

    for img_path in tqdm(img_list , desc ='') : 
        img = cv2.imread(img_path)
        file_name = img_path.split('/')[-1]
        

        windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, crop_size, 0)

        for window in windows:
            subset = img[ window.indices()]

            x_start = window.indices()[0].start
            y_start = window.indices()[1].start
            
            file_store_path = os.path.join(store_path, file_name)
            file_store_path = file_store_path.replace(f'.{source_ext}', f'_{x_start}_{y_start}_{crop_size}.{store_ext}')
            cv2.imwrite(file_store_path, subset)


if __name__ == '__main__':
    

    main()