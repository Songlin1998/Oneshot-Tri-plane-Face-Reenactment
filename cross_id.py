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
    ckpt = '.../checkpoint/003000.pt'
    gen = Multiview(ckpt,device)

    transform_256 = torchvision.transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    intrinsics = FOV_to_intrinsics(fov_degrees=18.837, device=device)
    
    # target
    target_video_path = '.../train/id10822#YYvD0PsyuJs#00001.txt#000.mp4'
    frames_paths = sorted(glob.glob(target_video_path + '/*.png'))
    labels_file = os.path.join(target_video_path,'dataset.json')
    with open(labels_file, 'r') as f:
        file = json.load(f)
    labels = file['labels']
    labels = [labels[i][1] for i in range(len(labels))]
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    img_target = Image.open(frames_paths[0]).convert('RGB')
    camera_target = torch.from_numpy(labels[0]).unsqueeze(0).to(device)
    img_target = transform_256(img_target).unsqueeze(0).to(device)
    
    # source
    source_video_path ='.../train/id10817#ZhBQlOryF2Y#00002.txt#000.mp4' 
    frames_paths = sorted(glob.glob(source_video_path + '/*.png'))
    img_source = Image.open(frames_paths[0]).convert('RGB')
    img_source = transform_256(img_source).unsqueeze(0).to(device)

    # cross identity
    img_driven = gen.multi_gen(img_source, img_target, camera_target)
    img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_source_save = (img_source.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_target_save = (img_target.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_save = torch.cat([img_source_save,img_target_save,img_driven],dim=2)
    PIL.Image.fromarray(img_save[0].cpu().numpy(), 'RGB').save(f'cross_id.png')

    # reconstrcution
    img_driven = gen.multi_gen(img_target, img_target, camera_target)
    img_driven = (img_driven.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # img_source_save = (img_source.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_target_save = (img_target.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_save = torch.cat([img_target_save,img_target_save,img_driven],dim=2)
    PIL.Image.fromarray(img_save[0].cpu().numpy(), 'RGB').save(f'reconstruction.png')

    # Multi-view generation: yaw
    imgs = []
    angle_p = -0.2
    for angle_y, angle_p in [(.4, angle_p), (.2, angle_p), (0, angle_p),(-.2, angle_p), (-.4, angle_p)]:
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        img = gen.multi_gen(img_source,img_target,camera_params)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
    img = torch.cat(imgs, dim=2)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'multiview_yaw.png')

    # Multi-view generation: pitch
    imgs = []
    angle_y = -0.2
    for angle_y, angle_p in [(angle_y,.4), (angle_y,.2), (angle_y, 0),(angle_y,-.2), (angle_y,-.4)]:
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        img = gen.multi_gen(img_source,img_target,camera_params)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
    img = torch.cat(imgs, dim=2)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'multiview_pitch.png')

if __name__ == "__main__":
    
    main()