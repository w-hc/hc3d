import json
import numpy as np
from numpy.linalg import inv
from pathlib import Path
import imageio
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
    # imgs, poses, render_poses, [H, W, f], i_split = load_llff()
    imgs, K, poses = load_blender("train")
    H, W = imgs[0].shape[:2]

    downsize_hw = 10
    imgs = batch_img_resize(imgs, downsize_hw)

    cam_locs = poses[:, :3, -1]
    radius = np.linalg.norm(cam_locs, axis=1)
    # print(f"scene radius {radius}")

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

    m = load_mesh()
    cones.append(m)

    for i in range(len(imgs)):
        po = poses[i]
        im = imgs[i]
        geom = generate_cam(po, [1, 0, 0], im)
        cones.extend(geom)

        # rays = shoot_rays(K, po)
        # cones.extend([rays])

    imgs, K, poses = load_blender("val")
    imgs = batch_img_resize(imgs, downsize_hw)
    for i in range(len(imgs)):
        po = poses[i]
        im = imgs[i]
        geom = generate_cam(po, [0, 1, 0], im)
        cones.extend(geom)

    imgs, K, poses = load_blender("test")
    imgs = batch_img_resize(imgs, downsize_hw)
    for i in range(len(imgs)):
        po = poses[i]
        im = imgs[i]
        geom = generate_cam(po, [0, 0, 1], im)
        cones.extend(geom)

    # for po in render_poses:
    #     geom = generate_cam(po, [0, 1, 0])
    #     cones.extend(geom)

    o3d.visualization.draw(cones, show_skybox=False)


# def load_llff():
#     from load_llff import load_llff_data
#     root = Path('./data/nerf_llff_data/fern')
#     images, poses, bds, render_poses, i_test = load_llff_data(
#         root, factor=8, recenter=True, bd_factor=0.75,
#         spherify=False, path_zflat=False
#     )
#     hwf = poses[0, :3, -1]
#     poses = poses[:, :3, :4]

#     if not isinstance(i_test, list):
#         i_test = [i_test]

#     llffhold = 8
#     if llffhold > 0:
#         print('Auto LLFF holdout,', llffhold)
#         i_test = np.arange(images.shape[0])[::llffhold]

#     i_val = i_test
#     i_train = np.array([
#         i for i in np.arange(int(images.shape[0]))
#         if (i not in i_test and i not in i_val)
#     ])

#     n = poses.shape[0]
#     full_poses = np.zeros((n, 4, 4))
#     full_poses[:, 3, 3] = 1
#     full_poses[:, :3, :4] = poses

#     return images, full_poses, render_poses, hwf, [i_train, i_val, i_test]


def blend_rgba(img):
    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
    return img


def load_blender(split, scene="lego", half_res=False):
    assert split in ("train", "val", "test")

    root = "/scratch"
    root = Path(root) / "nerf_data/nerf_synthetic" / scene

    with open(root / f'transforms_{split}.json', "r") as f:
        meta = json.load(f)

    imgs, poses = [], []

    for frame in meta['frames']:
        file_name = root / f"{frame['file_path']}.png"
        im = imageio.imread(file_name)
        c2w = frame['transform_matrix']

        imgs.append(im)
        poses.append(c2w)

    imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
    imgs = blend_rgba(imgs)
    poses = np.array(poses).astype(float)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    f = 1 / np.tan(camera_angle_x / 2) * (W / 2)

    if half_res:
        raise NotImplementedError()

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])  # note OpenGL -ve z convention;

    return imgs, K, poses


def load_mesh():
    p = Path("~/lego.ply").expanduser()
    p = str(p)
    mesh = o3d.io.read_triangle_mesh(p)
    # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # verts = mesh.vertex['positions'].numpy()
    # print(mesh)

    return mesh


if __name__ == "__main__":
    plot_inward_facing_views()
    # load_mesh()
