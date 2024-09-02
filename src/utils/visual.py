
def visual_project_points(points, mask=None):
    from PIL import Image
    import matplotlib.pyplot as plt
    points = rearrange(points, "1 f n c -> f n c")
    mask = rearrange(mask, "1 f n -> f n")
    f = points.shape[0]
    for i in range(f):
        plt.figure(figsize=(10, 10))
        point = points[i].reshape(-1, 2).cpu().numpy()
        tm_mask = mask[i].reshape(-1).cpu().numpy()
        plt.scatter(point[tm_mask == 1, 0], point[tm_mask == 1, 1], c="r", s=1)
        plt.scatter(point[tm_mask == 0, 0], point[tm_mask == 0, 1], c="b", s=1)
        plt.savefig(f"visual/visual_project_points_{i}.png")


def visual_camera(c2w, intrisics, idx=0, ray_orgins=None, ray_directions = None, points=None, sensor_size=1):
    from PIL import Image
    import matplotlib.pyplot as plt
    import pytransform3d.camera as pc
    import pytransform3d.transformations as pt
    import numpy as np
    from src.utils.project import ray_sample
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    intrisics = intrisics.reshape(-1, 3, 3)
    c2w = c2w.reshape(-1, 4, 4)
    num_frames = c2w.shape[0]
    virtual_image_distance = 1
    for i in range(0, num_frames):
        cam2world = c2w[i].cpu().numpy()
        intrinsic_matrix = intrisics[i].cpu().numpy()[:3, :3]
        ax = pt.plot_transform(ax=ax,A2B=cam2world, s=1, strict_check=False)
        pc.plot_camera(
            ax, cam2world=cam2world, M=intrinsic_matrix, sensor_size=np.array([sensor_size, sensor_size]),
            virtual_image_distance=virtual_image_distance, strict_check=False, alpha=1 if i == 0 else 0.5)
        
    # if ray_orgins is None or ray_directions is None:
    #     ray_orgins, ray_directions = ray_sample(c2w, intrisics[:, :3, :3], 4)
    # # plot rays
    # ray_orgins = ray_orgins.reshape(-1, 3).cpu().numpy()
    # ray_directions = ray_directions.reshape(-1, 3).cpu().numpy()
    # if points is not None:
    #     points = points.reshape(-1, 3).cpu().numpy()
    #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", s=2)
    # plt.quiver(ray_orgins[:, 0], ray_orgins[:, 1], ray_orgins[:, 2], ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2], length=1, normalize=True)
    # 绘制 [-1, 1] x [-1, 1] x[-1, 1]空间
    vertices = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [1, 1, -1]
    ]

    # Define the edges of the cube
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Plot the edges of the cube
    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0]]
        y = [vertices[edge[0]][1], vertices[edge[1]][1]]
        z = [vertices[edge[0]][2], vertices[edge[1]][2]]
        ax.plot(x, y, z, color='b')
    range_x = 1.5
    plt.xlim(-range_x, range_x)
    plt.ylim(-range_x, range_x)
    ax.set_zlim(-range_x, range_x)
    plt.savefig(f"visual/3d_pose_{idx}.png")

from PIL import Image

def concatenate_images(image_list, axis):
    # Assuming all images are the same size, get dimensions of first image
    w, h = image_list[0].size

    if axis == 'h':
        # Create a new image with width equal to total width of all images and height of the first image
        total_width = w * len(image_list)
        new_image = Image.new('RGB', (total_width, h))

        # Iterate through image list and paste each image into new image
        for index, image in enumerate(image_list):
            new_image.paste(image, (index * w, 0))

    elif axis == 'v':
        # Create a new image with height equal to total height of all images and width of the first image
        total_height = h * len(image_list)
        new_image = Image.new('RGB', (w, total_height))

        # Iterate through image list and paste each image into new image
        for index, image in enumerate(image_list):
            new_image.paste(image, (0, index * h))

    else:
        raise ValueError("Invalid axis. It should be 'h' or 'v'.")

    return new_image
