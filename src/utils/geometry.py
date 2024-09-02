import numpy as np
import torch
import torchvision
from einops import rearrange, repeat


def get_position_map_from_depth(depth, mask, intrinsics, extrinsics, image_wh=None):
    """Compute the position map from the depth map and the camera parameters for a batch of views.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        intrinsics (torch.Tensor): The camera intrinsics matrices with the shape (B, 3, 3).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    u_coord, v_coord = torch.meshgrid(
        torch.arange(image_wh[0]), torch.arange(image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map by back-projecting depth pixels to 3D space
    x = (
        (u_coord - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (v_coord - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_position_map(
    depth,
    cam2world_matrix,
    intrinsics,
    resolution,
    scale=0.001,
    offset=0.5,
):
    """
    Compute the position map from the depth map and the camera parameters for a batch of views.

    depth: (B, F, 1, H, W)
    cam2world_matrix: (B, 4, 4)
    intrinsics: (B, 3, 3)
    resolution: int
    """
    bsz = depth.shape[0]
    depths = rearrange(depth, "b f c h w -> (b f) h w c").to(
        dtype=cam2world_matrix.dtype
    )
    masks = depths > 0
    cam2world_matrices = rearrange(cam2world_matrix, "b f c1 c2 -> (b f) c1 c2")
    intrinsics = rearrange(intrinsics, "b f c1 c2 -> (b f) c1 c2")

    position_maps = get_position_map_from_depth(
        depths, masks, intrinsics, cam2world_matrices
    )

    # Convert to meters and clamp values
    position_maps = (position_maps * scale + offset).clamp(0.0, 1.0)

    position_maps = rearrange(position_maps, "(b f) h w c -> b f c h w", b=bsz)

    return position_maps
