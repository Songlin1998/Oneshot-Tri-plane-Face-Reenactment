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
from facenet_pytorch import InceptionResnetV1
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from training.dual_discriminator import DualDiscriminator_r1

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        
        
        # start_time = time.time()
        self.gen = Generator(size=args.size,plane_size=256, rendering_size = 64,style_dim=args.latent_dim_style,motion_dim=args.latent_dim_motion).to(device)
        
        self.superresolution = SuperresolutionHybrid4X(channels=32, img_resolution=args.size, 
                                                    sr_num_fp16_res=4, # Number of fp16 layers in superresolution
                                                    sr_antialias=True).to(device)
        
        ckpt = torch.load(args.resume_ckpt)
        self.gen.load_state_dict(ckpt["gen"])
        self.superresolution.load_state_dict(ckpt["superresolution"])

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.superresolution = DDP(self.superresolution, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
   

        param_groups = [
            {'params': self.gen.parameters()},
            {'params': self.superresolution.parameters()}
        ]

        self.g_optim = optim.Adam(
            param_groups,
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(rank)
        self.smoothl1_loss = torch.nn.L1Loss()
        self.deca = DECA(config = deca_cfg, device=rank).eval()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06)

    def gen_update_same_id(self, img_source_0, img_target_256, camera_target, img_target_128, nerf_size, mask_target):
        
        self.gen.train()
        self.gen.zero_grad()
        requires_grad(self.gen, True)

        self.superresolution.train()
        self.superresolution.zero_grad()
        requires_grad(self.superresolution, True)

        img_d, z, TVloss= self.gen(img_source_0, img_target_256, camera_target, nerf_size)
        ws = z.unsqueeze(1).to(img_d['image_128'].device)
        img_target_recon_256 = self.superresolution(img_d['image_128'], img_d['feature_image'], ws) # [32,3,256,256]

        img_target_256 = img_target_256.to(img_d['image_128'].device) # [32,3,256,256]
        img_target_128 = img_target_128.to(img_d['image_128'].device) 
        img_source_256 = img_source_0.to(img_d['image_128'].device) # [32,3,256,256]
        mask_target = mask_target.to(img_d['image_128'].device)

        vgg_loss_256 = self.criterion_vgg(img_target_recon_256, img_target_256).mean()
        l1_loss_256 = self.smoothl1_loss(img_target_256,img_target_recon_256)
        l1_loss_128 = self.smoothl1_loss(img_target_128,img_d['image_128'])
        
        z_recon = self.gen.module.E(img_target_recon_256)
        z_target = self.gen.module.E(img_target_256)
        id_loss_lia = self.smoothl1_loss(z_recon[:,:512],z[:,:512])
        exp_loss_lia = self.smoothl1_loss(z_recon[:,512:],z[:,512:])

        id_consist_loss = self.smoothl1_loss(z[:,:512],z_target[:,:512])

        mask_loss = self.smoothl1_loss(torch.mul(img_target_256,mask_target),torch.mul(img_target_recon_256,mask_target))
         
        g_loss =  0.02*vgg_loss_256 + 2000*mask_loss + 10*(l1_loss_256 + l1_loss_128) + 0.25*TVloss + 50*id_consist_loss
    
        g_loss.backward()
        self.g_optim.step()

        return vgg_loss_256, l1_loss_256, l1_loss_128, img_target_recon_256, id_loss_lia, exp_loss_lia, mask_loss, TVloss, id_consist_loss

    def gen_update_recon(self, img_source_0, img_target_256, camera_target, img_target_128, nerf_size, mask_target):
        
        self.gen.train()
        self.gen.zero_grad()
        requires_grad(self.gen, True)

        self.superresolution.train()
        self.superresolution.zero_grad()
        requires_grad(self.superresolution, True)

        img_d, z, TVloss= self.gen(img_target_256, img_target_256, camera_target, nerf_size)
        ws = z.unsqueeze(1).to(img_d['image_128'].device)
        img_target_recon_256 = self.superresolution(img_d['image_128'], img_d['feature_image'], ws) # [32,3,256,256]

        img_target_256 = img_target_256.to(img_d['image_128'].device) # [32,3,256,256]
        img_target_128 = img_target_128.to(img_d['image_128'].device) 
        mask_target = mask_target.to(img_d['image_128'].device)

        # encoding_loss = self.encoding_loss(z_id_c, z_id_all)
        vgg_loss_256 = self.criterion_vgg(img_target_recon_256, img_target_256).mean()
        l1_loss_256 = self.smoothl1_loss(img_target_256,img_target_recon_256)
        l1_loss_128 = self.smoothl1_loss(img_target_128,img_d['image_128'])
        
        z_recon = self.gen.module.E(img_target_recon_256)
        z_target = self.gen.module.E(img_target_256)
        id_loss_lia = self.smoothl1_loss(z_recon[:,:512],z[:,:512])
        exp_loss_lia = self.smoothl1_loss(z_recon[:,512:],z[:,512:])

        id_consist_loss = self.smoothl1_loss(z[:,:512],z_target[:,:512])

        mask_loss = self.smoothl1_loss(torch.mul(img_target_256,mask_target),torch.mul(img_target_recon_256,mask_target))
         
        g_loss = 0.02*vgg_loss_256 + 2000*mask_loss + 10*(l1_loss_256 + l1_loss_128) + 0.25*TVloss + 50*id_consist_loss
    
        g_loss.backward()
        self.g_optim.step()

        return vgg_loss_256, l1_loss_256, l1_loss_128, img_target_recon_256, id_loss_lia, exp_loss_lia, mask_loss, TVloss, id_consist_loss

    def gen_update_cross_id(self, img_source_0, img_target_256, camera_target, img_target_128, nerf_size):
        
        self.gen.train()
        self.gen.zero_grad()
        requires_grad(self.gen, True)

        self.superresolution.train()
        self.superresolution.zero_grad()
        requires_grad(self.superresolution, True)

      

        img_d, z, TVloss = self.gen(img_source_0, img_target_256, camera_target, nerf_size)
        ws = z.unsqueeze(1).to(img_d['image_128'].device)
        img_target_recon_256 = self.superresolution(img_d['image_128'], img_d['feature_image'], ws) # [32,3,256,256]

        img_target_256 = img_target_256.to(img_d['image_128'].device) # [32,3,256,256]
        img_target_128 = img_target_128.to(img_d['image_128'].device) 
        img_source_256 = img_source_0.to(img_d['image_128'].device) # [32,3,256,256]

        img_source_224 = torch.nn.functional.interpolate(img_source_256,[224,224])
        img_target_recon_224 = torch.nn.functional.interpolate(img_target_recon_256,[224,224])

        self.deca.zero_grad()
        requires_grad(self.deca, True)

        codedict = self.deca.encode(img_target_recon_224)
        codedict_id = self.deca.encode(img_source_224)
        id_loss = self.smoothl1_loss(torch.cat([codedict['shape'],codedict['tex'],codedict['light'].reshape(-1,27)],dim=1),
                                    torch.cat([codedict_id['shape'],codedict_id['tex'],codedict_id['light'].reshape(-1,27)],dim=1))

        z_recon = self.gen.module.E(img_target_recon_256)
        z_target = self.gen.module.E(img_target_256)
        id_loss_lia = self.smoothl1_loss(z_recon[:,:512],z[:,:512])
        exp_loss_lia = self.smoothl1_loss(z_recon[:,512:],z[:,512:])

        id_triplet_loss = self.triplet_loss(z[:,:512],z[:,:512],z_target[:,:512])
        g_loss = 10*id_loss_lia+ 0.25*TVloss + id_triplet_loss + 50*id_loss
        
        g_loss.backward()
        self.g_optim.step()

        return img_target_recon_256, id_loss_lia, exp_loss_lia, TVloss, id_triplet_loss, id_loss


    def sample(self, img_source_0, img_target_256, camera_target, img_target_128, nerf_size):
        self.gen.eval()
        self.superresolution.eval()
        with torch.no_grad():
            self.gen.eval()
            self.superresolution.eval()
            img_d, z, TVloss = self.gen(img_source_0, img_target_256, camera_target,nerf_size)
            ws = z.unsqueeze(1).to(img_d['image_128'].device)
            img_target_recon_256 = self.superresolution(img_d['image_128'], img_d['feature_image'], ws)

        return img_d, img_target_recon_256

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "superresolution": self.superresolution.module.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )