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
from tqdm import tqdm
import imageio

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

def imgs2gif(imgs_path):
    gif_name =os.path.join(imgs_path,'video.gif') 
    imgs = sorted(glob.glob(imgs_path + '/*.png'))
    frames = []
    for img in imgs:
        frames.append(imageio.imread(img))
    imageio.mimwrite(gif_name, frames, 'GIF', duration=0.04)


def main():

    device = 'cuda:0'
    
   
    ckpt = '.../001000.pt'
    gen = Multiview(ckpt,device)
    transform_256 = torchvision.transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    intrinsics = FOV_to_intrinsics(fov_degrees=18.837, device=device)
    
    source_video_path = '.../train/id10257#fI-1LgrsNNE#00005.txt#000.mp4'
    target_video_path = '.../train/id10822#YYvD0PsyuJs#00001.txt#000.mp4'
    frames_paths_source = sorted(glob.glob(source_video_path + '/*.png'))
    frames_paths_target = sorted(glob.glob(target_video_path + '/*.png'))

    source_frame = transform_256(Image.open(frames_paths_source[0]).convert('RGB')).unsqueeze(0).to(device)

    labels_file = os.path.join(target_video_path,'dataset.json')
    with open(labels_file, 'r') as f:
        file = json.load(f)
    labels = file['labels']
    labels = [labels[i][1] for i in range(len(labels))]
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    source_name = (source_video_path.split('/')[-1]).split('#')[0]
    target_name = (target_video_path.split('/')[-1]).split('#')[0]
    
    
    # video driven
    print('video driven')
    save_path = os.path.join('...', 'results',f'{source_name}_{target_name}','video_driven')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for target_idx in tqdm(range(50)):

        target_frame = transform_256(Image.open(frames_paths_target[target_idx]).convert('RGB')).unsqueeze(0).to(device)
        target_camera = torch.from_numpy(labels[target_idx]).unsqueeze(0).to(device)
        img_driven = gen.multi_gen(source_frame, target_frame, target_camera)
        img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        file_path =os.path.join(save_path,f'{target_idx}.png')
        PIL.Image.fromarray(img_driven[0].cpu().numpy(), 'RGB').save(file_path)
    
    imgs2gif(save_path)

    # multiview - 1
    print('multiview-1')
    angle_y = 0
    angle_p = 0
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    target_camera = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    save_path = os.path.join('...','results',f'{source_name}_{target_name}',f'multiview_1')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for target_idx in tqdm(range(25)):

        target_frame = transform_256(Image.open(frames_paths_target[target_idx]).convert('RGB')).unsqueeze(0).to(device)
        # target_camera = torch.from_numpy(labels[target_idx]).unsqueeze(0).to(device)
        img_driven = gen.multi_gen(source_frame, target_frame, target_camera)
        img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        file_path =os.path.join(save_path,f'{target_idx}.png')
        PIL.Image.fromarray(img_driven[0].cpu().numpy(), 'RGB').save(file_path)
    
    imgs2gif(save_path)

    # multiview - 2
    print('multiview-2')
    angle_y = -0.8
    angle_p = -0.2
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    target_camera = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    save_path = os.path.join('...','results',f'{source_name}_{target_name}',f'multiview_2')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for target_idx in tqdm(range(25)):

        target_frame = transform_256(Image.open(frames_paths_target[target_idx]).convert('RGB')).unsqueeze(0).to(device)
        # target_camera = torch.from_numpy(labels[target_idx]).unsqueeze(0).to(device)
        img_driven = gen.multi_gen(source_frame, target_frame, target_camera)
        img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        file_path =os.path.join(save_path,f'{target_idx}.png')
        PIL.Image.fromarray(img_driven[0].cpu().numpy(), 'RGB').save(file_path)

    imgs2gif(save_path)

    # multiview - 3
    print('multiview-3')
    angle_y = 0.8
    angle_p = -0.2
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    target_camera = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    save_path = os.path.join('...','results',f'{source_name}_{target_name}',f'multiview_3')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for target_idx in tqdm(range(25)):

        target_frame = transform_256(Image.open(frames_paths_target[target_idx]).convert('RGB')).unsqueeze(0).to(device)
        # target_camera = torch.from_numpy(labels[target_idx]).unsqueeze(0).to(device)
        img_driven = gen.multi_gen(source_frame, target_frame, target_camera)
        img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        file_path =os.path.join(save_path,f'{target_idx}.png')
        PIL.Image.fromarray(img_driven[0].cpu().numpy(), 'RGB').save(file_path)

    imgs2gif(save_path)

if __name__ == "__main__":
    
    main()