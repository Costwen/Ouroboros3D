import torch
import numpy as np
def get_intrinsics_by_fov(fov, image_size=1):
    focal_length = 0.5 * image_size / np.tan(0.5 * fov)
    intrinsics = np.array(
        [
            [focal_length, 0, image_size/2],
            [0, focal_length, image_size/2],
            [0, 0, 1],
        ],
        dtype=np.float32
    )
    intrinsics = torch.from_numpy(intrinsics)
    intrinsics_4x4 = torch.zeros(4, 4)
    intrinsics_4x4[:3, :3] = intrinsics
    intrinsics_4x4[3, 3] = 1.0  
    return intrinsics_4x4