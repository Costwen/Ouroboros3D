import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .options import Options


def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-8) + 1e-8)

class GaussianRenderer:
    def __init__(self):
        fovy: float = 49.1
        znear: float = 0.5
        zfar: float = 2.5

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        self.proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
        self.proj_matrix[2, 3] = 1
        self.output_size: int = 512
        
    def _render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]
        bg_color = torch.ones(3, dtype=torch.float32, device=device)
        # loop of loop...
        images = []
        alphas = []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.output_size,
                    image_width=self.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.output_size, self.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.output_size, self.output_size)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }
   
    def render(self, gaussians, viewpoint_cameras, B, V, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        # loop of loop...
        images = []
        alphas = []
        depths = []
        original_dtype = torch.get_default_dtype()  # 保存原始dtype
        torch.set_default_dtype(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            for b in range(B):
                # pos, opacity, scale, rotation, shs
                means3D = gaussians[b, :, 0:3].contiguous().float()
                opacity = gaussians[b, :, 3:4].contiguous().float()
                scales = gaussians[b, :, 4:7].contiguous().float()
                rotations = gaussians[b, :, 7:11].contiguous().float()
                rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

                for v in range(V):
                    viewpoint_camera = viewpoint_cameras[b][v]
                    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
                    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)
                    dtype = torch.float32
                    viewmatrix = viewpoint_camera.world_to_camera.to(device).to(dtype)
                    projmatrix = viewpoint_camera.full_projection.to(device).to(dtype)
                    campos = viewpoint_camera.camera_center.to(device).to(dtype)

                    raster_settings = GaussianRasterizationSettings(
                        image_height=int(viewpoint_camera.height),
                        image_width=int(viewpoint_camera.width),
                        tanfovx=tanfovx,
                        tanfovy=tanfovy,
                        bg=bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=viewmatrix,
                        projmatrix=projmatrix,
                        sh_degree=0,
                        campos=campos,
                        prefiltered=False,
                        debug=False,
                    )

                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                    # Rasterize visible Gaussians to image, obtain their radii (on screen).
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None
                    )

                    rendered_image = rendered_image.clamp(0, 1)

                    images.append(rendered_image)
                    alphas.append(rendered_alpha)
                    depths.append(rendered_depth)
                    
        torch.set_default_dtype(original_dtype) 
        images = torch.stack(images, dim=0).view(B, V, 3, int(viewpoint_camera.height), int(viewpoint_camera.width))
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, int(viewpoint_camera.height), int(viewpoint_camera.width))
        depths = torch.stack(depths, dim=0).view(B, V, 1, int(viewpoint_camera.height), int(viewpoint_camera.width))
        return {
            "image": images.to(original_dtype), # [B, V, 3, H, W]
            "alpha": alphas.to(original_dtype), # [B, V, 1, H, W]
            "depth": depths.to(original_dtype) # [B, V, 1, H, W]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            # opacity = kiui.op.inverse_sigmoid(opacity)
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians