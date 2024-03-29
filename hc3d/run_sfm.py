from pathlib import Path
import json
import numpy as np
from numpy.linalg import inv, svd, det, norm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image

import open3d as o3d

from .vis import quick_vis_3d, o3d_pc, draw_camera
from .mpl_interactive import Visualizer as TwoViewVis
from .sfm import (
    t_and_R_from_pose_pair,
    essential_from_t_and_R,
    F_from_K_and_E,
    E_from_K_and_F,
    t_and_R_from_essential,
    disambiguate_four_chirality_by_triangulation,
    triangulate,
    normalized_eight_point_algorithm,
    eight_point_algorithm,
    bundle_adjustment,
    align_B_to_A
)

from .render import (
    compute_intrinsics, compute_extrinsics, as_homogeneous, homogenize
)

DATA_ROOT = Path("./data")


def load_sfm_data():
    root = DATA_ROOT / "sfm"
    poses = np.load(root / "camera_poses.npy")
    visibility = np.load(root / "multiview_visibility.npy")
    pts_3d = np.load(root / "pts_3d.npy")
    with open(root / "camera_intrinsics.json", "r") as f:
        intr = json.load(f)
    img_w, img_h, fov = intr['img_w'], intr['img_h'], intr['vertical_fov']

    data = SimpleNamespace(
        img_w=img_w, img_h=img_h, fov=fov,
        poses=poses, visibility=visibility, pts_3d=pts_3d
    )
    return data


def read_view_image(i):
    fname = DATA_ROOT / "sfm" / f"view_{i}.png"
    img = np.array(Image.open(fname))
    return img


def common_visible_points(data, view_indices):
    """view_indices: a list of view indices e.g. [0, 1, 4]
    """
    mask = data.visibility[view_indices].all(axis=0)
    common_pts = data.pts_3d[mask]
    return common_pts


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


class Problems():
    def __init__(self):
        data = load_sfm_data()
        self.view1, self.view2 = 1, 5
        pose1 = data.poses[self.view1]
        pose2 = data.poses[self.view2]
        pts_3d = common_visible_points(data, [self.view1, self.view2])

        img_w, img_h = data.img_w, data.img_h
        fov = data.fov
        K = compute_intrinsics(img_w / img_h, fov, img_h)

        self.data = data
        self.K = K
        self.pose1 = pose1
        self.pose2 = pose2
        self.pts_3d = pts_3d
        self.img_w, self.img_h = img_w, img_h

    def q2(self):
        pts_3d = self.pts_3d
        K = self.K
        pose1, pose2 = self.pose1, self.pose2
        img_w, img_h = self.img_w, self.img_h

        fname = str(DATA_ROOT / "optimus_prime.obj")
        mesh = o3d.io.read_triangle_mesh(fname, False)

        quick_vis_3d(
            mesh, o3d_pc(pts_3d),
            draw_camera(K, pose1, img_w, img_h, scale=10),
            draw_camera(K, pose2, img_w, img_h, scale=10),
        )

    def q4(self):
        pts_3d = self.pts_3d
        K = self.K
        pose1, pose2 = self.pose1, self.pose2

        img1 = read_view_image(self.view1)
        img2 = read_view_image(self.view2)

        x1s = project(pts_3d, K, pose1)
        x2s = project(pts_3d, K, pose2)

        t, R = t_and_R_from_pose_pair(pose1, pose2)
        K = K[:3, :3]
        E = essential_from_t_and_R(t, R)
        F = F_from_K_and_E(K, E)

        # now let's test the the epipolary geometry
        on_epi_line = np.einsum("np, pq, nq -> n", x2s, F, x1s)
        assert np.allclose(on_epi_line, 0)  # alright

        # epipole on img2
        c1_on_img2 = K @ t
        c1_on_img2 = homogenize(c1_on_img2)
        assert np.allclose(c1_on_img2.T @ F, 0)

        # epipole on img1
        c2_on_img1 = K @ (-inv(R) @ t)
        c2_on_img1 = homogenize(c2_on_img1)
        assert np.allclose(F @ c2_on_img1, 0)

        print(c1_on_img2)
        print(c2_on_img1)

        vis = TwoViewVis()
        vis.vis(img1, img2, F)

    def q10(self):
        self.sfm_pipeline(draw_config=True, use_BA=False, final_vis=True)

    def q12(self):
        self.sfm_pipeline(use_noise=True, use_BA=True, final_vis=True)

    def sfm_pipeline(self, use_noise=False, use_BA=False, draw_config=False, final_vis=False):
        pts_3d = self.pts_3d
        K = self.K
        img_w, img_h = self.img_w, self.img_h
        pose1, pose2 = self.pose1, self.pose2

        x1s = project(pts_3d, K, pose1)
        x2s = project(pts_3d, K, pose2)

        if use_noise:
            x1s, x2s = corruption_pipeline(x1s, x2s)

        full_K = K
        K = K[:3, :3]

        F = normalized_eight_point_algorithm(x1s, x2s, img_w, img_h)
        # F = eight_point_algorithm(x1s, x2s)

        E = E_from_K_and_F(K, F)
        four_tR_hypothesis = t_and_R_from_essential(E)

        p1, p2, t, R = disambiguate_four_chirality_by_triangulation(
            four_tR_hypothesis, x1s, x2s, full_K, draw_config=draw_config
        )
        pred_pts = triangulate(full_K @ inv(p1), x1s, full_K @ inv(p2), x2s)

        if use_BA:
            p1, p2, pred_pts = bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts)

        pred_pts, p1, p2 = align_B_to_A(pred_pts, p1, p2, pts_3d)

        if not use_noise:
            assert np.allclose(pred_pts, pts_3d)

        if final_vis:
            red = (1, 0, 0)
            green = (0, 1, 0)
            blue = (0, 0, 1)

            quick_vis_3d(
                o3d_pc(pts_3d, red),
                o3d_pc(pred_pts, green),
                draw_camera(K, pose1, img_w, img_h, 10, red),
                draw_camera(K, pose2, img_w, img_h, 10, red),
                draw_camera(K, p1, img_w, img_h, 10, blue),
                draw_camera(K, p2, img_w, img_h, 10, blue),
            )

            # quick_vis_3d(
            #     o3d_pc(pred_pts, green),
            #     draw_camera(K, p1, img_w, img_h, 10, blue),
            #     draw_camera(K, p2, img_w, img_h, 10, blue),
            # )

    def teaser(self):
        K = self.K

        n = len(self.data.poses)
        img_w, img_h = self.img_w, self.img_h

        fname = str(DATA_ROOT / "optimus_prime.obj")
        mesh = o3d.io.read_triangle_mesh(fname, False)

        cams = [
            draw_camera(K, self.data.poses[i], img_w, img_h, scale=10)
            for i in range(n)
        ]

        quick_vis_3d(mesh, o3d_pc(self.data.pts_3d), *cams)

    def show_visib(self):
        plt.imshow(self.data.visibility, aspect="auto", interpolation='nearest')
        plt.xlabel("3D points")
        plt.ylabel("Camera view")
        plt.yticks(np.arange(10, dtype=int))
        plt.show()


def corruption_pipeline(x1s, x2s):
    s = 1
    noise = s * np.random.randn(*x1s.shape)
    noise[:, -1] = 0  # cannot add noise to the 1! fatal error

    x1s = x1s + noise
    x2s = x2s + noise

    x1s = flip_correspondence(x1s, 0.02)
    x1s = np.rint(x1s)
    x2s = np.rint(x2s)
    return x1s, x2s


def flip_correspondence(x1s, perc):
    n = x1s.shape[0]
    num_wrong = int(n * perc)
    chosen = np.random.choice(n, size=num_wrong, replace=False)
    x1s[chosen] = x1s[np.random.permutation(chosen)]
    return x1s


def main():
    engine = Problems()
    engine.teaser()
    engine.show_visib()
    # engine.q2()
    # engine.q4()
    engine.q10()
    # engine.q12()


if __name__ == "__main__":
    main()
