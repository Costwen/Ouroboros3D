import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .unet import UNet
from .options import Options
from .gs import GaussianRenderer
from einops import rearrange
from src.utils.project import ray_sample, encode_plucker
import torchvision.transforms.functional as TF
class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
        num_frames: int,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.output_channels = 14
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=num_frames,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer()

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x) * 0.7 if num_frames > 4 else 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again
        self.num_frames = num_frames


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, c2w, fov, elevation=0):
        
        from kiui.cam import orbit_camera
        from src.utils.project import get_rays
        b, f, _, _ = c2w.shape

        c2w = rearrange(c2w, "b f h w -> (b f) h w")
        fov = rearrange(fov, "b f -> (b f)")
        
        rays_embeddings = []
        for i in range(c2w.shape[0]):

            rays_o, rays_d = get_rays(c2w[i].reshape(4, 4).float(), self.opt.input_size, self.opt.input_size, fov[i].item(), opengl=False)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(c2w.device) # [V, 6, h, w]

        # rays_embeddings = rearrange(rays_embeddings, "(b f) c h w -> b f c h w", b=b, f=f)
        
        return rays_embeddings
        

    def forward_gaussians(self, images, timesteps, batch):
        # return: Gaussians: [B, dim_t]
        b, v, c, h, w = images.shape
        dtype = images.dtype
        c2w = batch['c2w']
        intrinsics = batch['intrinsics']
        fov = batch['fov']
        images = 0.5*images + 0.5
        if v > self.num_frames:
            # uniform sampling
            selected_indices = torch.arange(0, v, v//self.num_frames)
            # Generate 4 unique random indices from the range of views
            # selected_indices = torch.randperm(v)[:4]
            # Select the views and corresponding parameters based on the indices
            images = images[:, selected_indices, :, :, :]
            c2w = c2w[:, selected_indices] # b f 4 4
            intrinsics = intrinsics[:, selected_indices]
            fov = fov[:, selected_indices]
        # save iamges for visualization
        
        plucker_emebdding = self.prepare_default_rays(c2w, fov) # [V, 6, h, w]
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        images = rearrange(images, "b v c h w -> (b v) c h w")
        images = TF.normalize(
            images,
            mean=IMAGENET_DEFAULT_MEAN, 
            std=IMAGENET_DEFAULT_STD
        )
        images = F.interpolate(images, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        
        images = torch.concat([images, plucker_emebdding], dim=1)
        # upsample
        # scale_factor = 4
        # images = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        images = images.to(dtype)
        x = self.unet(images, timesteps) 
        x = self.conv(x) # [B*4, 14, h, w]

        # x = x.reshape(b, v, 14, self.opt.splat_size, self.opt.splat_size)

        # x = x.permute(0, 1, 3, 4, 2).reshape(b, -1, 14)
        
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v = 4)

        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

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
        results['depths_pred'] = results['depth']
        results['gaussians'] = gaussians

        return results
