import numpy as np
import pandas as pd
import imageio
import os
import subprocess
import warnings
import glob
import time
from argparse import ArgumentParser
from skimage.transform import resize
import cv2
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")


def process(videos_path,save_path):

    for video_path in tqdm(videos_path):

        video_name = os.path.basename(video_path).split('.')[0]
        imgs_save_path = os.path.join(save_path,video_name)
        if not os.path.exists(imgs_save_path):
            os.makedirs(imgs_save_path)

        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        while(True):
            _, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame,(256,256))
            cv2.imwrite(os.path.join(imgs_save_path, f'{frame_num}' + '.png'), frame)
            frame_num = frame_num + 1
        cap.release()

if __name__ == "__main__":
    
    videos_folder = '...'
    videos = sorted(glob.glob(videos_folder+'/*.mp4'))

    train_videos = videos[:35600] # 35600
    test_videos = videos[35600:]

    print('train_length:',len(train_videos))
    print('test_length:',len(test_videos))

    train_save_path = '.../data/celebv/train'
    test_save_path =  '.../data/celebv/test'

    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    process(videos_path=train_videos,save_path=train_save_path)
    process(videos_path=test_videos,save_path=test_save_path)

