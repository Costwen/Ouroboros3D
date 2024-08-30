import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .unet import UNet
from .options import Options
from .gs import GaussianRenderer
from einops import rearrange
from src.utils.project import ray_sample, encode_plucker

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.output_channels = 12
        self.unet = UNet(
            10, self.output_channels,  # 1 + 1 + 3 + 4 + 3
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention
        )

        # last conv
        self.conv = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer()

        # activations...
        self.depth_act = lambda x: torch.sigmoid(x)
        # self.scale_act = lambda x: 0.01 * F.softplus(x)
        self.scale_act = lambda x: F.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again
        self.near = 1
        self.far = 2
        self.scale_min = 0
        self.scale_max = 2 / 512 * 8


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images, timesteps, batch):
        # return: Gaussians: [B, dim_t]
        b, v, c, h, w = images.shape
        c2w = batch['c2w']
        intrinsics = batch['intrinsics']
        
        if v > 4:
            # uniform sampling
            selected_indices = torch.arange(0, v, v//4) 
            # Generate 4 unique random indices from the range of views
            # selected_indices = torch.randperm(v)[:4]
            # Select the views and corresponding parameters based on the indices
            images = images[:, selected_indices, :, :, :]
            c2w = c2w[:, selected_indices]
            intrinsics = intrinsics[:, selected_indices]

        ray_origins, ray_dirs = ray_sample(c2w.reshape(-1, 4, 4), intrinsics.reshape(-1, 3, 3), resolution = h)
        plucker_emebdding = encode_plucker(ray_origins, ray_dirs)
        plucker_emebdding = rearrange(plucker_emebdding, "bv (h w) c -> bv c h w", h =h, w = w)
        images = rearrange(images, "b v c h w -> (b v) c h w")
        images = torch.concat([images, plucker_emebdding], dim=1)
        # upsample
        scale_factor = 4
        images = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        x = self.unet(images, timesteps) 
        x = self.conv(x) # [B*4, 14, h, w]

        ray_origins, ray_dirs = ray_sample(c2w.reshape(-1, 4, 4), intrinsics.reshape(-1, 3, 3), resolution = int(x.shape[-1]))
        # x = x.reshape(b, v, 14, self.opt.splat_size, self.opt.splat_size)
        # x = x.permute(0, 1, 3, 4, 2).reshape(b, -1, 14)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", v = 4)
        ray_origins = rearrange(ray_origins, "(b v) hw c -> b (v hw) c", v = 4)
        ray_dirs = rearrange(ray_dirs, "(b v) hw c -> b (v hw) c", v = 4)
        
        depths = self.depth_act(x[..., :1]) # [B, N, 1]
        depths = self.near + (self.far - self.near) * depths
        # add x and y offset
        pos = ray_origins + ray_dirs * depths

        # visual pos 3d
        # from src.utils.visual import visual_camera
        # c2w = batch['c2w'] # b n i j
        # intrinsics = batch['intrinsics']

        # visual_camera(
        #     c2w, intrinsics, 
        #     points=pos
        # )
        # exit()

        # opacity = self.opacity_act(x[..., 3:4])
        # scale = self.scale_act(x[..., 4:7])
        # rotation = self.rot_act(x[..., 7:11])
        # rgbs = self.rgb_act(x[..., 11:])

        opacity = self.opacity_act(x[..., 1:2])
        scale = self.scale_act(x[..., 2:5])
        scale = self.scale_min + (self.scale_max - self.scale_min) * scale
        rotation = self.rot_act(x[..., 5:9])
        rgbs = self.rgb_act(x[..., 9:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        return gaussians

    
    def forward(self, images, batch, timesteps, step_ratio=1):
        # data: output of the dataloader
        # return: loss
        results = {}
        loss = 0
        b, v, c, h, w = images.shape
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images, timesteps, batch)
        cameras = batch['cameras']

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, cameras, b, v, bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['gaussians'] = gaussians


        return results
