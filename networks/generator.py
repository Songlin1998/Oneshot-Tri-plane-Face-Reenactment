from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis
from .encoder import Encoder_exp, Encoder_id, Fusion
from training.networks_stylegan2 import SynthesisNetwork
from networks.styledecoder import Motion_Field
from networks.styledecoder import Warping
from networks.styledecoder import Plane_motion_field
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.renderer import ImportanceRenderer
import torch
from training.networks_stylegan2 import FullyConnectedLayer
from training.superresolution import SuperresolutionHybrid4X
import time
import pickle


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class Volume_Rendering(nn.Module):
    def __init__(self, img_resolution=256, neural_rendering_resolution=64):
        super(Volume_Rendering, self).__init__()
        # self.neural_rendering_resolution = neural_rendering_resolution
        self.ray_sampler = RaySampler()
        self.renderer = ImportanceRenderer()
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32})
        
        self.rendering_kwargs =  {
                                    'disparity_space_sampling': False,
                                    'clamp_mode': 'softplus',
                                    'depth_resolution': 48, # number of uniform samples to take per ray.
                                    'depth_resolution_importance': 48, # number of importance samples to take per ray.
                                    'ray_start': 2.25, # near point along each ray to start taking samples.
                                    'ray_end': 3.3, # far point along each ray to stop taking samples. 
                                    'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
                                    'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
                                    'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
                                }

        self.canonical_plane = nn.Parameter(torch.randn(1,3,32,256,256))

    def forward(self, c, planes, nerf_size):
        
        # ws = torch.cat([z_id,z_exp],dim=1).unsqueeze(1)
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, nerf_size)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes += self.canonical_plane.repeat(len(planes),1,1,1,1) # 学习id残差

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, 
                                                                        ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = nerf_size
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        feature_image = torch.nn.functional.interpolate(feature_image,[128,128])
        depth_image = torch.nn.functional.interpolate(depth_image,[128,128])

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        # sr_image = self.superresolution(rgb_image, feature_image, ws)

        return {'image_128': rgb_image, 'image_depth': depth_image, 'feature_image': feature_image}

    def sample_mixed(self, coordinates, directions, planes):
        
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

class Generator(nn.Module):
    def __init__(self, size=256, plane_size=256, rendering_size = 64,nerf_channels=96, style_dim=512, motion_dim=512):
        super(Generator, self).__init__()

        # encoder
        self.E = Encoder_id(size=256,dim=512*2)
        # input: [bs, 3, 256, 256] output: [bs, 512]
        
        # stylegan generator → tri-plane_canonical
        self.G_tri_plane = SynthesisNetwork(w_dim=style_dim, img_resolution=size, img_channels=32*3)
        
        # input: [bs, 10, 512] output: [bs, 3*32, 256, 256]

        # 产生基于id和exp调制的deformation plane
        self.motion_field = Plane_motion_field(style_dim=style_dim, 
                                            plane_channels=nerf_channels, 
                                            plane_size=plane_size, # plane的尺寸
                                            motion_num=motion_dim)
        # input：[bs, 512], [bs, 512] output:[bs, 96, 256, 256]

        # volume rendering
        self.render = Volume_Rendering(img_resolution=size, neural_rendering_resolution=rendering_size) # 中间image输出的尺寸
        
        # input: z_id[bs,512], z_exp[bs,512], c[bs,25], planes[bs,96,256,256]
        # output: [bs, 3, 256, 256]

    def forward(self, img_source_0, img_target_256, camera_target,nerf_size):
        
        bs = img_source_0.shape[0]
        imgs = torch.cat([img_source_0, img_target_256],dim=0)
        z = self.E(imgs)
        z_id = z[:bs,:512]
        # z_exp = z[bs:,512:]
    
        plane_id = self.G_tri_plane(z_id.unsqueeze(1).repeat(1,14,1)) # [bs,96,256,256]
        plane_exp, z_exp = self.motion_field(z[:bs,512:],z[bs:,512:])

        # 渲染
        plane = plane_id + plane_exp
        img_d = self.render(camera_target,plane,nerf_size)

        # density regularization
        ws = torch.cat([z_id,z_exp],dim=1)
        initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 0.004
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.render.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), plane.view(len(plane), 3, 32, plane.shape[-2], plane.shape[-1]))['sigma']
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]
        TVloss = torch.nn.functional.mse_loss(sigma_initial, sigma_perturbed)

        return img_d, torch.cat([z_id,z_exp],dim=1), TVloss