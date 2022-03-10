import json
from pathlib import Path
import numpy as np
from numpy.linalg import inv, svd, det, norm
import open3d as o3d
from PIL import Image

import trimesh
from tqdm import tqdm

from .vis import OpenVisWrapper, generate_spiral_camera_trajectory, draw_camera
from .render import (
    compute_intrinsics,
    compute_extrinsics,
    camera_pose,
    as_homogeneous,
    homogenize,
    rays_through_pixels
)


N_SAMPLES = 10000


def load():
    fname = "./data/optimus_prime.obj"
    mesh = o3d.io.read_triangle_mesh(fname, False)
    tri = trimesh.load(fname, force='mesh', skip_texture=True)

    # sample a point cloud, and sort the points by coordinates
    pc = mesh.sample_points_uniformly(N_SAMPLES)
    pts = np.asarray(pc.points)
    inds = np.lexsort(pts.T)
    pts = pts[inds]

    return mesh, tri, pts


def process():
    mesh, tmesh, pts_3d = load()

    img_w, img_h = 1600, 1200
    fov = 60
    K = compute_intrinsics(img_w / img_h, fov, img_h)
    traj_poses = generate_spiral_camera_trajectory(
        pts_3d, num_steps=10, num_rounds=2, height_multi=2, radius_mult=1.
    )
    cam_geoms = [draw_camera(K, pose, img_w, img_h, scale=10)
                 for pose in traj_poses]
    pts_3d = as_homogeneous(pts_3d)

    visibility = []
    for i in tqdm(range(len(traj_poses))):
        # gradually remove those not visible
        candidates = np.arange(N_SAMPLES, dtype=int)

        pose = traj_poses[i]
        E = inv(pose)
        P = K @ E

        pts_2d = homogenize(pts_3d @ P.T)[:, :2]
        inbound = in_img_bound(pts_2d, img_w, img_h)
        # inbound = np.arange(N_SAMPLES, dtype=int)
        pts_2d = pts_2d[inbound]
        candidates = candidates[inbound]

        rays = rays_through_pixels(K, pts_2d)
        rays = rays @ pose.T

        rays = rays[:, :3]
        origin = pose[:3, -1].reshape(1, -1)
        origin = origin.repeat(len(rays), axis=0)

        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            tmesh, scale_to_box=False)
        # m = intersector.intersects_any(origin, rays)
        locs, r_inds, t_inds = intersector.intersects_location(
            origin, rays, multiple_hits=False)

        assert (r_inds == np.sort(r_inds)).all()  # don't remove this
        candidates = candidates[r_inds]
        # assert (r_inds == np.arange(N_SAMPLES)).all()  # maybe there can be no hit?

        dist = norm((locs - pts_3d[candidates, :3]), axis=1)
        threshold = 1e-4
        hit_mask = (dist < threshold)
        candidates = candidates[hit_mask]

        tqdm.write(f"{len(candidates)} points in view")

        visible_mask = np.zeros(N_SAMPLES, dtype=bool)
        visible_mask[candidates] = True
        visibility.append(visible_mask)

    del visible_mask

    visibility = np.stack(visibility, axis=0)
    coverage = visibility.any(axis=0)
    print(f"point coverage rate {coverage.sum() / N_SAMPLES:.3f}")

    # remove points that are completely invisible from outer views
    pts_3d = pts_3d[coverage]
    visibility = visibility[:, coverage]
    assert visibility.any(axis=0).all()
    print(f"{len(pts_3d)} samples in total")

    del coverage

    export_data(img_w, img_h, K, traj_poses, mesh, pts_3d, visibility)
    # visual(img_w, img_h, o3d_pc(pts_3d[:, :3]), *cam_geoms)


def export_data(img_w, img_h, K, traj_poses, mesh, pts_3d, visibility):
    root = Path("./sfm")
    poses = []
    n = len(traj_poses)
    renderer = OpenVisWrapper(img_w, img_h)
    renderer.add(mesh)
    renderer.add(o3d_pc(pts_3d[:, :3]))
    renderer.opts.point_size = 3.

    for i in tqdm(range(n)):
        pose = traj_poses[i]
        E = inv(pose)
        P = K @ E
        pts = pts_3d[visibility[i]]
        xys = homogenize(pts @ P.T)[:, :2]
        renderer.set_intrinsics(K)
        renderer.set_extrinsics(E)

        poses.append(pose)

        img = renderer.to_image()

        # manually mark key points; confirmed it's aligned, but it doesn't look nice
        # xys = np.rint(xys).astype(int)
        # xs, ys = xys.T
        # img[ys, xs] = (1, 0, 0)

        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(root / f"view_{i}.png")

    with open(root / "camera_intrinsics.json", "w") as f:
        json.dump({
            "img_w": img_w,
            "img_h": img_h,
            "vertical_fov": 60,
        }, f)

    np.save(root / "camera_poses.npy", np.asarray(traj_poses))
    np.save(root / "pts_3d.npy", pts_3d)
    np.save(root / "multiview_visibility.npy", visibility)


def check_visibility():
    import matplotlib.pyplot as plt
    visibility = np.load("visible.npy")
    plt.imshow(visibility, aspect="auto")
    plt.show()


def in_img_bound(pts_2d, img_w, img_h):
    xs, ys = pts_2d.T
    mask = (xs > -0.5) & (xs < img_w - 0.5) & (ys > -0.5) & (ys < img_h - 0.5)
    return mask


def visual(img_w, img_h, *geoms):
    renderer = OpenVisWrapper(img_w, img_h)
    for obj in geoms:
        renderer.add(obj)
    renderer.blocking_run()


def o3d_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def main():
    process()


if __name__ == "__main__":
    main()
    # check_visibility()


"""
def load_horse_pointcloud():
    root = Path("../hw2/data")
    fname = str(root / "horse.obj")
    mesh = o3d.io.read_triangle_mesh(fname, True)
    # sample a point cloud from the surface of the mesh
    pcloud = mesh.sample_points_uniformly(250)
    points = np.asarray(pcloud.points)
    return points


def vis_horse(fov):
    img_w, img_h = 1600, 1200
    eye = np.array([0, 0, 3])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])

    mesh = load_horse_mesh()
    renderer = OpenVisWrapper(img_w, img_h)
    renderer.add(mesh)

    K = compute_intrinsics(img_w / img_h, fov, img_h)
    E = compute_extrinsics(eye, front, up)
    # renderer.set_intrinsics(K)
    # renderer.set_extrinsics(E)
    renderer.blocking_run()


def load_horse_mesh():
    fname = "/Users/haochen/Downloads/ding-food-vessel-11th-10th-century-bce/source/Mia_000816_Ding_64k/Mia_000816_Ding_64k.obj"
    fname = "/Users/haochen/Downloads/optimus-prime/source/op_test_01/op_test_01.obj"
    fname = "/Users/haochen/Downloads/source/OO_Raiser.dae"
    fname = "/Users/haochen/Downloads/fun/tri.obj"
    fname = "/Users/haochen/Downloads/optimus_prime/scene.gltf"
    fname = "/Users/haochen/Downloads/fun/this.stl"
    # fname = "/Users/haochen/Downloads/gg.obj"
    # fname = "/Users/haochen/Downloads/ttt/untitled.obj"

    # root = Path("/Users/haochen/work/3d_model_zoo")
    # fname = str(root / "spot.obj")

    # fname = "/Users/haochen/Downloads/ccc.ply"
    mesh = o3d.io.read_triangle_mesh(fname, True)
    pc = mesh.sample_points_uniformly(50000)
    return pc


def batch_nullspace(xs):
    # batched qr is only supported on numpy >= 1.22; it's a fairly new feature
    # since we are in 3d, just use the skew matrix for now;
    # xs: [n, d]
    # for each elem in the batch, find the (d - 1) orthogonal directions
    # return shape [n, d - 1, d]
    xs = xs[..., np.newaxis]
    Q, R = np.linalg.qr(xs, mode="complete")
    print(Q.shape)


def pose1():
    eye = np.array([0, 0, 2])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])
    pose = camera_pose(eye, front, up)
    return pose


def pose2():
    eye = np.array([2, 0, 0.5])
    front = np.array([-1, 0, 0])
    up = np.array([0, 1, 0])
    pose = camera_pose(eye, front, up)
    return pose


def test_triangulation_limit():
    # [Np, Nv * 2, 4]
    constraint = np.random.rand(50000, 25 * 2, 4)
    U, s, V_t = svd(constraint, full_matrices=False)
    pts = V_t[:, -1]
    pts = homogenize(pts)
    print(pts.shape)


def iterative_triangulate():
    from tqdm import tqdm
    constraint = np.random.rand(50000, 6 * 2, 4)
    for i in tqdm(range(50000)):
        cons = constraint[i]
        U, s, V_t = svd(cons, full_matrices=False)
        pts = V_t[:, -1]

"""
