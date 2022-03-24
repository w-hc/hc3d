import numpy as np
from numpy.linalg import inv
from pathlib import Path
from load_blender import load_blender_data
from load_llff import load_llff_data
import open3d as o3d

from hc3d.vis import CameraCone
from hc3d.render import compute_intrinsics, unproject
from hc3d.utils import batch_img_resize


def get_K(H=500, W=500, fov=60):
    K = compute_intrinsics(W / H, fov, H)
    return K


def shoot_rays(K, pose):
    h = 200
    pixs = np.array([
        [10, h],
        [200, h],
        [400, h]
    ])
    pts = unproject(K, pixs, depth=1.0)
    pts = np.concatenate([
        pts,
        np.array([0, 0, 0, 1]).reshape(1, -1),
    ], axis=0)  # origin, followed by 4 img corners
    pts = pts @ pose.T
    pts = pts[:, :3]
    pts = pts.astype(np.float32)

    n = len(pixs)
    lines = np.array([
        [i, n] for i in range(n)
    ], dtype=np.int32)

    color = [1, 1, 0]
    colors = np.array([color] * len(lines), dtype=np.float32)

    lset = o3d.t.geometry.LineSet()
    lset.point['positions'] = pts
    lset.line['indices'] = lines
    lset.line['colors'] = colors

    return lset


def test_rays(H, W, K):
    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32), indexing='xy'
    )
    xys = np.stack([xs, ys], axis=-1)
    my_rays = unproject(K, xys.reshape(-1, 2))
    my_rays = my_rays.reshape(int(H), int(W), 4)[:, :, :3]
    return


def plot_inward_facing_views():
    imgs, poses, render_poses, [H, W, f], i_split = load_llff()
    print("data loaded")

    train_inds, val_inds, test_inds = i_split

    cam_locs = poses[:, :3, -1]
    radius = np.linalg.norm(cam_locs, axis=1)

    imgs = batch_img_resize(imgs, 10)

    K = np.array([
        [f, 0, -W / 2],
        [0, -f, -H / 2],
        [0, 0, -1]
    ])

    # test_rays(H, W, K)

    # K = get_K(H, W, 50)
    # NeRF blender actually follows OpenGL camera convention (except top-left corner); nice
    # but its world coordinate is z up. I find it strange.

    def generate_cam(po, color, im=None):
        cone = CameraCone(K, po, W, H, scale=0.3,
                          top_left_corner=(0, 0), color=color)
        lset = cone.as_line_set()
        if im is None:
            return [lset]
        else:
            # o3d img tsr requires contiguous array
            im = np.ascontiguousarray(im)
            view_plane = cone.as_view_plane(im)
            return [lset, view_plane]

    cones = []

    for inx in train_inds:
        po = poses[inx]
        im = imgs[inx]
        geom = generate_cam(po, [1, 0, 0], im)
        cones.extend(geom)

        rays = shoot_rays(K, po)
        cones.extend([rays])

    # for inx in val_inds:
    #     po = poses[inx]
    #     im = imgs[inx]
    #     geom = generate_cam(po, [0, 1, 0], im)
    #     cones.extend(geom)

    # for inx in test_inds:
    #     po = poses[inx]
    #     im = imgs[inx]
    #     geom = generate_cam(po, [0, 0, 1], im)
    #     cones.extend(geom)

    # for po in render_poses:
    #     geom = generate_cam(po, [0, 1, 0])
    #     cones.extend(geom)

    o3d.visualization.draw(cones, show_skybox=False)


def load_blender():
    # root = Path('./data/nerf_synthetic/lego')
    root = Path('/scratch/nerf_data/nerf_synthetic/lego')
    imgs, poses, render_poses, hwf, i_split = \
        load_blender_data(root, half_res=False)
    render_poses = render_poses.numpy()
    return imgs, poses, render_poses, hwf, i_split


def load_llff():
    root = Path('./data/nerf_llff_data/fern')
    images, poses, bds, render_poses, i_test = load_llff_data(
        root, factor=8, recenter=True, bd_factor=0.75,
        spherify=False, path_zflat=False
    )
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]

    if not isinstance(i_test, list):
        i_test = [i_test]

    llffhold = 8
    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]

    i_val = i_test
    i_train = np.array([
        i for i in np.arange(int(images.shape[0]))
        if (i not in i_test and i not in i_val)
    ])

    n = poses.shape[0]
    full_poses = np.zeros((n, 4, 4))
    full_poses[:, 3, 3] = 1
    full_poses[:, :3, :4] = poses

    return images, full_poses, render_poses, hwf, [i_train, i_val, i_test]


if __name__ == "__main__":
    plot_inward_facing_views()
