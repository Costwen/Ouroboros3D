from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import copy
import numpy as np
from kiui.op import safe_normalize

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def compute_rel_transform(c2w, ray_origins, ray_directions):
    """
    c2w: (b x f) x 4 x 4
    ray_origins: (b x f) x (h x w) x 3
    ray_directions: (b x f) x (h x w) x 3
    return:
        canonical_w2c: (b x f) x (h x w) x 3 x 4
    """
    upv = c2w[:, :3, 1]  # (b x f) x 3
    upv = rearrange(upv, "bf i -> bf 1 i")  # (b x f) x 1 x 3
    rdotup = (ray_directions * upv).sum(-1, keepdims=True)  # (b x f) x (h x w) x 1
    orthoup = upv - rdotup * ray_directions  # 史密斯正交化
    orthoup = F.normalize(orthoup, dim=-1)  # (b x f) x (h x w) x 3
    vec0 = torch.cross(orthoup, ray_directions)  # (b x f) x (h x w) x 3
    vec0 = F.normalize(vec0, dim=-1)  # (b x f) x (h x w) x 3
    r_relatives = torch.stack(
        (vec0, orthoup, ray_directions), dim=-1
    )  # (b x f) x (h x w) x 3 x 3
    r_relatives_T = r_relatives.transpose(-1, -2)  # (b x f) x (h x w) x 3 x 3
    translation_relative = torch.einsum(
        "... i j, ... j -> ... i", -r_relatives_T, ray_origins
    ).unsqueeze(
        -1
    )  # (b x f) x (h x w) x 3
    canonical_w2c = torch.cat(
        (r_relatives_T, translation_relative), dim=-1
    )  # (b x f) x (h x w) x 3 x 4
    return canonical_w2c


def ray_sample(cam2world_matrix, intrinsics, resolution, sensor_size=1, normalize=True):
    """
    Create batches of rays and return origins and directions.

    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    resolution: int

    ray_origins: (N, M, 3)
    ray_dirs: (N, M, 3)
    """
    
    N, M = cam2world_matrix.shape[0], resolution**2
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]
    uv = torch.stack(
        torch.meshgrid(
            torch.arange(
                resolution, dtype=torch.float32, device=cam2world_matrix.device
            ),
            torch.arange(
                resolution, dtype=torch.float32, device=cam2world_matrix.device
            ),
            indexing="ij",
        )
    ) 

    uv = uv * (1.0 * sensor_size / resolution) + (0.5 * sensor_size / resolution)


    uv = repeat(uv, "c h w -> b (h w) c", b=N)
    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = torch.ones((N, M), device=cam2world_matrix.device)

    x_lift = (
        (
            x_cam
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z_cam
    )
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack(
        (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1
    ).to(cam2world_matrix.dtype)
    world_rel_points = torch.bmm(
        cam2world_matrix, cam_rel_points.permute(0, 2, 1)
    ).permute(0, 2, 1)[:, :, :3]

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    if normalize:
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs


def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def sample_stratified(
    ray_origins,
    ray_start,
    ray_end,
    depth_resolution,
    disparity_space_sampling=False,
    return_depth=False,
):
    """
    Return depths of approximately uniformly spaced samples along rays.
    """
    N, M, _ = ray_origins.shape
    if disparity_space_sampling:
        depths_coarse = (
            torch.linspace(0, 1, depth_resolution, device=ray_origins.device)
            .reshape(1, 1, depth_resolution, 1)
            .repeat(N, M, 1, 1)
        )
        if not return_depth:
            depth_delta = 1 / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
        depths_coarse = 1.0 / (
            1.0 / ray_start * (1.0 - depths_coarse) + 1.0 / ray_end * depths_coarse
        )
    else:
        if type(ray_start) == torch.Tensor:
            depths_coarse = linspace(ray_start, ray_end, depth_resolution).permute(
                1, 2, 0, 3
            )
            if not return_depth:
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
        else:
            depths_coarse = (
                torch.linspace(
                    ray_start, ray_end, depth_resolution, device=ray_origins.device
                )
                .reshape(1, 1, depth_resolution, 1)
                .repeat(N, M, 1, 1)
            )
            if not return_depth:
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta
    return depths_coarse


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)

    bb_min = [
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
    ]
    bb_max = [
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
    ]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[
        ..., 1
    ]
    tymax = (
        bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]
    ) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[
        ..., 2
    ]
    tzmax = (
        bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]
    ) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def sample_points(ray_origins, ray_directions, rendering_options, return_depth=False):
    if rendering_options["ray_start"] == rendering_options["ray_end"] == "auto":
        ray_start, ray_end = get_ray_limits_box(
            ray_origins, ray_directions, box_side_length=rendering_options["box_warp"]
        )
        is_ray_valid = ray_end > ray_start
        if torch.any(is_ray_valid).item():
            ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        depths_coarse = sample_stratified(
            ray_origins,
            ray_start,
            ray_end,
            rendering_options["n_samples"],
            rendering_options["disparity_space_sampling"],
            return_depth=return_depth,
        )
    else:
        # Create stratified depth samples
        depths_coarse = sample_stratified(
            ray_origins,
            rendering_options["ray_start"],
            rendering_options["ray_end"],
            rendering_options["n_samples"],
            rendering_options["disparity_space_sampling"],
            return_depth=return_depth,
        )
    batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
    # Coarse Pass
    sample_coordinates = ray_origins.unsqueeze(
        -2
    ) + depths_coarse * ray_directions.unsqueeze(-2)
    sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1)
    return sample_coordinates, sample_directions, depths_coarse


def compute_projections(points, k_nearest_cameras):
    """
    points: (B, F, (H x W), N_samples, 3)
    k_nearest_cameras: (B, F, K, 32)
    return:
        pixel_locations: (B, F, K, (H x W), N_samples, 2)
        mask: (B, F, K, (H x W), N_samples)
    """
    b, f, k, _ = k_nearest_cameras.shape
    intrinsics = k_nearest_cameras[..., :16].reshape(b, f, k, 4, 4)  # (B, F, K, 4, 4)
    c2w = k_nearest_cameras[..., 16:].reshape(b, f, k, 4, 4)  # (B, F, K, 4, 4)
    points_h = torch.cat(
        [points, torch.ones_like(points[..., :1])], dim=-1
    )  # (B, F, (H x W), N_samples, 4)

    points_h = repeat(points_h, "b f n s c -> b f k n s c", k=k)
    w2p = torch.matmul(intrinsics, torch.inverse(c2w))  # (B, F, K, 4, 4)
    projections = torch.einsum("b f k i j, b f k n s j -> b f k n s i", w2p, points_h)

    # TODO: check is this correct
    pixel_locations = projections[..., :2] / torch.clamp(
        projections[..., 2:3], min=1e-8
    )
    pixel_locations = torch.clamp(
        pixel_locations, min=-10, max=10
    )  # to avoid grid sample nan
    mask = projections[..., 2] > 0  # opencv camera
    inbound = (
        (pixel_locations[..., 0] <= 1.0)
        & (pixel_locations[..., 0] >= 0)
        & (pixel_locations[..., 1] <= 1.0)
        & (pixel_locations[..., 1] >= 0)
    )
    mask = mask & inbound
    return pixel_locations, mask


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.
        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )


def encode_plucker(ray_origins, ray_dirs, harmonic_embedding=None):
    """
    ray to plucker w/ pos encoding
    """
    dtype = ray_origins.dtype
    ray_origins = ray_origins.to(torch.float32)
    ray_dirs = ray_dirs.to(torch.float32)
    plucker = torch.cat((torch.cross(ray_origins, ray_dirs, dim=-1), ray_dirs), dim=-1)
    if harmonic_embedding:
        plucker = harmonic_embedding(plucker)
    return plucker.to(dtype)


def process_cameras(
    cameras, select_ids=None, harmonic_embedding=None, rendering_options=None
):
    """
    cameras: (B, F, 32)
    select_ids: (B, F, K)
    return:
        sample_coordinates: (B, F, (H x W), N_samples, 3)
        sample_directions: (B, F, (H x W), N_samples, 3)
        query_plucker: (B, F, K, (H x W), 78), the plucker coordinates of the sampled rays
        reference_plucker: (B, F, K, (H x W), N_samples, 78), the plucker coordinates of the sampled points
    """
    _rendering_options = copy.deepcopy(rendering_options)
    _rendering_options["n_samples"] = min(_rendering_options["n_samples"], _rendering_options["resolution"]
    )
    resolution = _rendering_options["resolution"]
    n_samples = _rendering_options["n_samples"]
    b, f, c = cameras.shape
    if select_ids is None:
        select_ids = torch.arange(f, device=cameras.device)  # (k)
        select_ids = repeat(select_ids, "k -> b f k", b=b, f=f)
    k_views = select_ids.shape[2]

    full_cameras = repeat(cameras, "b k c -> b f k c", f=f)
    k_nearest_cameras = torch.zeros(b, f, k_views, c, device=cameras.device)
    b_ids = torch.arange(b)[:, None, None].expand(-1, f, k_views).to(cameras.device)
    f_ids = torch.arange(f)[None, :, None].expand(b, -1, k_views).to(cameras.device)
    # Use advanced indexing to get k nearest cameras
    k_nearest_cameras = full_cameras[b_ids, f_ids, select_ids]  # (B, F, K, 32)

    instrinsics, c2w = cameras[:, :, :16], cameras[:, :, 16:]
    c2w = rearrange(c2w, "b f (i j) -> (b f) i j", i=4, j=4)
    instrinsics = rearrange(instrinsics, "b f (i j) -> (b f) i j", i=4, j=4)[:, :3, :3]
    ray_origins, ray_directions = ray_sample(c2w, instrinsics, resolution)

    sample_coordinates, sample_directions, sample_depths = sample_points(
        ray_origins, ray_directions, _rendering_options, return_depth=True
    )
    sample_coordinates = rearrange(
        sample_coordinates, "(b f) n s c -> b f n s c", f=f, n=resolution**2
    )  # c = 3
    sample_depths = rearrange(
        sample_depths, "(b f) n s c -> b f n s c", f=f, n=resolution**2
    )  # c = 1
    sample_depths = (sample_depths - _rendering_options["ray_start"]) / (
        _rendering_options["ray_end"] - _rendering_options["ray_start"]
    )
    projected_points, mask = compute_projections(
        sample_coordinates, k_nearest_cameras
    )  # compute the k views projection

    query_plucker = encode_plucker(
        ray_origins, ray_directions, harmonic_embedding
    )  # (B F) N 78

    query_depth = harmonic_embedding(sample_depths)  # B F N S 78

    query_plucker = repeat(
        query_plucker, "(b f) n d -> b f k n s d", f=f, k=k_views, s=n_samples
    )  # B F K N 78
    query_depth = repeat(query_depth, "b f n s d -> b f k n s d", k=k_views)

    query_plucker = torch.cat(
        (query_plucker, query_depth), dim=-1
    )  # B F K N S 156 # 在 i in F 帧下的深度

    k_nearest_c2w = rearrange(
        k_nearest_cameras[..., 16:], "b f k (i j) -> b f k i j", i=4, j=4
    )
    origins_cam = repeat(
        k_nearest_c2w[..., :3, 3],
        "b f k i -> b f k n s i",
        n=resolution**2,
        s=n_samples,
    )
    sample_coordinates = repeat(
        sample_coordinates, "b f n s c -> b f k n s c", k=k_views
    )
    input_dirs = sample_coordinates - origins_cam  # (B F K N S 3)
    input_dirs = F.normalize(input_dirs, dim=-1)

    canonical_w2c = compute_rel_transform(c2w, ray_origins, ray_directions)
    canonical_w2c = repeat(
        canonical_w2c, "(b f) n i j -> b f k n s i j", f=f, s=n_samples, k=k_views
    )
    origins_cam = torch.cat(
        (origins_cam, torch.ones_like(origins_cam[..., :1])), dim=-1
    )
    cannonical_dirs = torch.einsum(
        "... i j, ... j-> ... i", canonical_w2c[..., :3, :3], input_dirs
    )
    cannonical_cam = torch.einsum("... i j, ... j -> ... i", canonical_w2c, origins_cam)

    reference_plucker = encode_plucker(
        cannonical_cam, cannonical_dirs, harmonic_embedding
    )  # depth of the s-th point in the k-th view
    reference_depth = harmonic_embedding(sample_depths)  # B F N S 78
    full_depth = repeat(reference_depth, "b k ... -> b f k ...", f=f)
    k_nearest_depth = full_depth[b_ids, f_ids, select_ids]
    reference_plucker = torch.cat(
        (reference_plucker, k_nearest_depth), dim=-1
    )  # B F K N S 156
    projected_points = projected_points * 2 - 1  # to [-1, 1]
    # return projected_points, mask, (ray_origins, ray_directions), (cannonical_cam, cannonical_dirs), sample_depths
    return projected_points, mask, query_plucker, reference_plucker, sample_depths
