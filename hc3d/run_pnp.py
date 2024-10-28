from pathlib import Path
import numpy as np
import open3d as o3d
from .render import (
    compute_intrinsics,
    compute_extrinsics,
    as_homogeneous,
    homogenize
)


def load_horse_pointcloud():
    root = Path("./data")
    fname = str(root / "horse.obj")
    mesh = o3d.io.read_triangle_mesh(fname, True)
    # sample a point cloud from the surface of the mesh
    pcloud = mesh.sample_points_uniformly(500)
    points = np.asarray(pcloud.points)
    return points


def ortho(xs):
    # for 3d, cross-product is good.
    # for d > 3, use qr factorization. much easier to write this way
    # xs: [n, 3]
    n, _ = xs.shape
    x1, x2, x3 = xs.T
    zeros = np.zeros(n)
    cross_product_mats = np.array([
        [zeros, -x3, x2],
        [x3, zeros, -x1],
        [-x2, x1, zeros]
    ])
    cross_product_mats = np.transpose(cross_product_mats, (2, 0, 1))
    return cross_product_mats


def pnp_localization(pts_2d, pts_3d):
    n = pts_2d.shape[0]

    pts_2d = ortho(pts_2d)  # [n, 3, 3]
    A = np.einsum("nab, nc -> nabc", pts_2d, pts_3d)
    A = A.reshape(3 * n, -1)

    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    P = L.reshape(3, 4)
    return P


def _pnp_calibration(pts_2d, pts_3d):
    n = pts_2d.shape[0]

    # this is based on the slides and books. But it's hard to remember
    A = []
    for i in range(n):
        u, v, _ = pts_2d[i]
        x, y, z, _ = pts_3d[i]
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])

    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    P = L.reshape(3, 4)
    return P


def main():
    pts_3d = load_horse_pointcloud()
    pts_3d = as_homogeneous(pts_3d)

    img_w, img_h = 600, 450
    fov = 90
    eye = np.array([2, 0, 0.5])
    front = np.array([-1, 0, 0])
    up = np.array([0, 1, 0])
    K = compute_intrinsics(img_w / img_h, fov, img_h)
    E = compute_extrinsics(eye, front, up)
    P = K @ E
    P = P / P[2, 3]

    pts_2d = pts_3d @ P.T
    pts_2d = homogenize(pts_2d)

    pred_P = pnp_localization(pts_2d, pts_3d)

    assert np.allclose(P, pred_P)


if __name__ == "__main__":
    main()
