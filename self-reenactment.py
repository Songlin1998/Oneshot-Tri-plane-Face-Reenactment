import torch
from networks.discriminator import Discriminator
from networks.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle
from training.superresolution import SuperresolutionHybrid4X
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from camera_utils import GaussianCameraPoseSampler
import math
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import numpy as np
import PIL
import glob
import json
import random

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Multiview(nn.Module):
    def __init__(self, pre_trained_ckpt, device):
        super(Multiview, self).__init__()

        # start_time = time.time()
        self.gen = Generator(size=256 ,plane_size=256, rendering_size = 64,style_dim=512,motion_dim=20).to(device)
        
        self.superresolution = SuperresolutionHybrid4X(channels=32, img_resolution=256, 
                                                    sr_num_fp16_res=4, # Number of fp16 layers in superresolution
                                                    sr_antialias=True).to(device)
        
        print("load model:", pre_trained_ckpt)
        ckpt = torch.load(pre_trained_ckpt)
        self.gen.load_state_dict(ckpt["gen"],strict=True)
        self.superresolution.load_state_dict(ckpt["superresolution"],strict=True)
        self.gen.eval()
        self.superresolution.eval()
        
    def multi_gen(self, img_source, img_target, camera_target):
        
        with torch.no_grad():
            img_d, z, _ = self.gen(img_source, img_target, camera_target, 64)
            ws = z.unsqueeze(1).to(img_d['image_128'].device)
            img_target_recon_256 = self.superresolution(img_d['image_128'], img_d['feature_image'], ws)
                
        return img_target_recon_256


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)

def main():

    device = 'cuda:0'
    
    # multiview generation
    ckpt = '.../008000.pt'
    target_video_path = '.../train/id10283#arklnCzCq48#00003.txt#001.mp4'
    source_video_path ='.../train/id10283#arklnCzCq48#00003.txt#001.mp4'
    target_idx = 446
    source_idx = 470

    gen = Multiview(ckpt,device)

    transform_256 = torchvision.transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    intrinsics = FOV_to_intrinsics(fov_degrees=18.837, device=device)
    
    # target
    frames_paths = sorted(glob.glob(target_video_path + '/*.png'))
    labels_file = os.path.join(target_video_path,'dataset.json')
    with open(labels_file, 'r') as f:
        file = json.load(f)
    labels = file['labels']
    labels = [labels[i][1] for i in range(len(labels))]
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    img_target = Image.open(frames_paths[target_idx]).convert('RGB')
    camera_target = torch.from_numpy(labels[target_idx]).unsqueeze(0).to(device)
    img_target = transform_256(img_target).unsqueeze(0).to(device)
    
    # source
    frames_paths = sorted(glob.glob(source_video_path + '/*.png'))
    img_source = Image.open(frames_paths[source_idx]).convert('RGB')
    img_source = transform_256(img_source).unsqueeze(0).to(device)

    # cross identity
    img_driven = gen.multi_gen(img_source, img_target, camera_target)
    img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_source_save = (img_source.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_target_save = (img_target.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img_source_save[0].cpu().numpy(), 'RGB').save(f'same_source.png')
    PIL.Image.fromarray(img_target_save[0].cpu().numpy(), 'RGB').save(f'same_driving.png')
    PIL.Image.fromarray(img_driven[0].cpu().numpy(), 'RGB').save(f'same_driven.png')

if __name__ == "__main__":
    
    main()