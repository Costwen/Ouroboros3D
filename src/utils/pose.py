# Adapted from  https://github.com/liuyuan-pal/SyncDreamer/blob/main/ldm/base_utils.py
import numpy as np
import cv2
import torch


def get_opencv_from_blender(c2w):
    c2w[:, :, 1:3] *= -1
    return c2w


def get_3x4_RT_matrix_from_blender(c2w):
    R, t = c2w[:, :3], c2w[:, 3]

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)
    return RT


def pose_inverse(pose):
    R = pose[:, :3].T
    t = -R @ pose[:, 3:]
    return np.concatenate([R, t], -1)


def project_points(pts, RT, K):
    pts = np.matmul(pts, RT[:, :3].transpose()) + RT[:, 3:].transpose()
    pts = np.matmul(pts, K.transpose())
    dpt = pts[:, 2]
    mask0 = (np.abs(dpt) < 1e-4) & (np.abs(dpt) > 0)
    if np.sum(mask0) > 0:
        dpt[mask0] = 1e-4
    mask1 = (np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1) > 0:
        dpt[mask1] = -1e-4
    pts2d = pts[:, :2] / dpt[:, None]
    return pts2d, dpt


def draw_keypoints(img, kps, colors=None, radius=2):
    out_img = img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color = [int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, -1)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0, 255, 0), -1)
    return out_img


def mask_depth_to_pts(mask, depth, K, rgb=None):
    hs, ws = np.nonzero(mask)
    depth = depth[hs, ws]
    pts = np.asarray([ws, hs, depth], np.float32).transpose()
    pts[:, :2] *= pts[:, 2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs, ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())


def transform_points_pose(pts, pose):
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape) == 1:
        return (R @ pts[:, None] + t[:, None])[:, 0]
    return pts @ R.T + t[None, :]


def pose_apply(pose, pts):
    return transform_points_pose(pts, pose)


def get_k_near_views(
    elevations,
    azimuths,
    k_near_views,
    num_views,
    add_global_k=None,
):
    k_near_views = k_near_views + add_global_k if add_global_k else k_near_views
    views = torch.cat((elevations.unsqueeze(1), azimuths.unsqueeze(1)), dim=1)
    distances = torch.cdist(views, views)
    torch.fill_(distances.diagonal(), 0.0)
    k_nearest_indices = torch.topk(distances, k_near_views, largest=False).indices

    if add_global_k is not None:
        global_part = (
            torch.arange(num_views - add_global_k, num_views)
            .unsqueeze(0)
            .repeat(elevations.shape[0], 1)
        )
        k_nearest_indices[:-add_global_k, -add_global_k:] = global_part[
            :-add_global_k, :add_global_k
        ]

    return k_nearest_indices
