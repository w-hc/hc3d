import numpy as np
from numpy.linalg import inv, svd, det, norm

import torch
import torch.optim as optim
from lietorch import SE3
from scipy.spatial.transform import Rotation as rot

from .vis import vis_3d, o3d_pc, draw_camera
from .render import (
    as_homogeneous,
    homogenize
)

def skew(xs):
    # xs: [n, 3]
    n, _ = xs.shape
    x1, x2, x3 = xs.T
    zeros = np.zeros(n)
    mats = np.array([
        [zeros, -x3, x2],
        [x3, zeros, -x1],
        [-x2, x1, zeros]
    ])
    mats = np.transpose(mats, (2, 0, 1))
    return mats


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


def pose_to_se3_embed(pose):
    R, t = pose[:3, :3], pose[:3, -1]
    tau = t
    phi = rot.from_matrix(R).as_quat()
    embed = np.concatenate([tau, phi], axis=0)
    embed = torch.as_tensor(embed)
    return embed


def as_torch_tensor(*args):
    return [torch.as_tensor(elem) for elem in args]


def torch_project(pts_3d, K, se3_pose):
    P = K @ se3_pose.inv().matrix()
    x1 = pts_3d @ P.T
    x1 = homogenize(x1)
    return x1


def bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts):
    embed1 = pose_to_se3_embed(p1)
    embed2 = pose_to_se3_embed(p2)

    # embed1 = torch.randn(7, dtype=torch.float64)
    # embed1[3:] = embed1[3:] / torch.norm(embed1[3:])
    # embed2 = torch.randn(7, dtype=torch.float64)
    # embed2[3:] = embed2[3:] / torch.norm(embed2[3:])

    embed1.requires_grad_(True)
    embed2.requires_grad_(True)

    x1s, x2s, full_K, pred_pts = \
        as_torch_tensor(x1s, x2s, full_K, pred_pts)

    pred_pts.requires_grad_(True)

    lr = 1e-3
    # optimizer = optim.SGD([embed1, embed2, pred_pts], lr=lr, momentum=0.9)
    optimizer = optim.Adam([embed1, embed2, pred_pts], lr=lr)

    n_steps = 10000
    for i in range(n_steps):
        optimizer.zero_grad()

        p1 = SE3.InitFromVec(embed1)
        p2 = SE3.InitFromVec(embed2)
        # p1 = SE3.Random([], dtype=torch.float64)
        # p2 = SE3.Random([], dtype=torch.float64)

        x1_hat = torch_project(pred_pts, full_K, p1)
        x2_hat = torch_project(pred_pts, full_K, p2)
        err1 = torch.norm((x1_hat - x1s), dim=1)
        err1 = err1.mean()
        err2 = torch.norm((x2_hat - x2s), dim=1)
        err2 = err2.mean()
        err = (err1 + err2) / 2

        err.backward()
        # don't give gradient to the homogeneous 1. This is always error-prone
        pred_pts.grad[:, -1] = 0  # adam might be able to handle this kind of perverse overparam.
        optimizer.step()

        if (i % (n_steps // 10)) == 0:
            print(f"step {i}, err: {err.item()}")

    p1 = SE3.InitFromVec(embed1).matrix().detach().numpy()
    p2 = SE3.InitFromVec(embed2).matrix().detach().numpy()
    pred_pts = pred_pts.detach().numpy()
    return p1, p2, pred_pts


def eight_point_algorithm(x1s, x2s):
    # estimate the fundamental matrix
    n, _ = x1s.shape
    cons = np.einsum("np, nq -> npq", x2s, x1s)
    cons = cons.reshape(n, -1)
    U, s, V_t = svd(cons, full_matrices=False)

    # cond number is the ratio of top rank / last rank singular values
    # when you solve Ax = b, you take s[0] / s[-1]. But in vision,
    # we convert the problem above into the form Ax = 0 s.t. ||x|| = 1
    # The nullspace is reserved for the solution.
    # This is called a "homogeneous system of equations" in linear algebra. This might be the reason
    # why homogeneous coordiantes are called homogeneous.
    # hence s[0] / s[-2].
    cond = s[0] / s[-2]
    print(f"condition number {cond}")

    F = V_t[-1].reshape(3, 3)
    F = F / F[2, 2]

    F = enforce_rank_2(F)
    return F


def enforce_rank_2(F):
    U, s, V_t = svd(F)
    s[-1] = 0
    F = U @ np.diag(s) @ V_t
    return F


def normalized_eight_point_algorithm(x1s, x2s, img_w, img_h):
    trans = np.array([
        [1, 0, -img_w / 2],
        [0, 1, -img_h / 2],
        [0, 0, 1]
    ])
    scale = np.array([
        [2 / img_w, 0, 0],
        [0, 2 / img_h, 0],
        [0, 0, 1]
    ])
    Q = scale @ trans

    x1s = x1s @ Q.T
    x2s = x2s @ Q.T
    F = eight_point_algorithm(x1s, x2s)
    F = Q.T @ F @ Q
    return F


def align_B_to_A(B, p1, p2, A):
    # B, A: [n, 3]
    assert B.shape == A.shape
    A = A[:, :3]
    B = B[:, :3]
    p1 = p1.copy()
    p2 = p2.copy()

    a_centroid = A.mean(axis=0)
    b_centroid = B.mean(axis=0)

    A = A - a_centroid
    B = B - b_centroid
    p1[:3, -1] -= b_centroid
    p2[:3, -1] -= b_centroid

    centroid = np.array([0, 0, 0])
    # root mean squre from centroid
    scale_a = (norm((A - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    scale_b = (norm((B - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    rms_ratio = scale_a / scale_b

    B = B * rms_ratio
    p1[:3, -1] *= rms_ratio
    p2[:3, -1] *= rms_ratio

    U, s, V_t = svd(B.T @ A)
    R = U @ V_t
    assert np.allclose(det(R), 1), "not special orthogonal matrix"
    new_B = B @ R  # note that here there's no need to transpose R... lol... this is subtle
    p1[:3] = R.T @ p1[:3]
    p2[:3] = R.T @ p2[:3]

    new_B = new_B + a_centroid
    new_B = as_homogeneous(new_B)
    p1[:3, -1] += a_centroid
    p2[:3, -1] += a_centroid
    return new_B, p1, p2


def triangulate(P1, x1s, P2, x2s):
    # x1s: [n, 3]

    # note that numpy's svd supports batch processing; don't have to do for loop

    # but in the general, large-scale setting, where there's visibility problem over lots of views
    # triangulation might have to to be done per-track, one-by-one.
    # otherwise you need batched svd on a [Np, Nv * 2, 4] tensor where Np is # of imgs, Nv # of views
    # it's very large and very sparse
    # note it's 2 not 3 because at its core each view contributes 2 constraints (a line constraint)

    assert x1s.shape == x2s.shape
    n = len(x1s)
    x1s = skew(x1s)  # [n, 3, 3 ]
    x2s = skew(x2s)

    constraint = np.stack([x1s @ P1, x2s @ P2], axis=1)
    constraint = constraint.reshape(n, -1, 4)
    U, s, V_t = svd(constraint, full_matrices=False)
    pts = V_t[:, -1]
    pts = homogenize(pts)
    return pts


def t_and_R_from_pose_pair(p1, p2):
    """the R and t that transforms points from pose 1's local frame to pose 2's local frame
    """
    T1to2 = inv(p2) @ p1
    R = T1to2[:3, :3]
    t = T1to2[:3, -1]
    return t, R


def pose_pair_from_t_and_R(t, R):
    """since we only have their relative orientation, the first pose
    is fixed to be identity
    """
    p1 = np.eye(4)
    T = np.block([
        [R, t.reshape(-1, 1)],
        [0, 0, 0, 1]
    ])
    p2 = inv(T @ inv(p1))
    return p1, p2


def essential_from_t_and_R(t, R):
    t_mat = skew(t.reshape(1, -1))[0]
    E = t_mat @ R
    return E


def t_and_R_from_essential(E):
    """this has even more ambiguity. there are 4 compatible (t, R) configurations
    out of which only 1 places all points in front of both cameras

    That the rank-deficiency in E induces 2 valid R is subtle;
    Longuet-Higgins came up with this at a time when SVD is not popularly known; incredible
    """
    U, s, V_t = svd(E)
    t = U[:, -1]  # last column of U
    t_mat = skew(t.reshape(1, -1))[0]

    # now solve procrustes to get back R
    U, s, V_t = svd(t_mat.T @ E)

    R = U @ V_t
    R1 = R * det(R)

    U[:, 2] = -U[:, 2]
    R = U @ V_t
    R2 = R * det(R)

    del R

    four_hypothesis = [
        [ t, R1],
        [-t, R1],
        [ t, R2],
        [-t, R2],
    ]
    return four_hypothesis


def disambiguate_four_chirality_by_triangulation(four, x1s, x2s, full_K, draw_config=False):
    # note that our camera is pointing towards its negative z axis
    num_infront = np.array([0, 0, 0, 0])
    four_pose_pairs = []

    for i, (t, R) in enumerate(four):
        # relative orientation is a fundamental ambiguity; just assume the pose is identity
        p1, p2 = pose_pair_from_t_and_R(t, R)
        pts = triangulate(full_K @ inv(p1), x1s, full_K @ inv(p2), x2s)
        nv1 = ((pts @ inv(p1).T)[:, 2] < 0).sum()
        nv2 = ((pts @ inv(p2).T)[:, 2] < 0).sum()

        num_infront[i] = nv1 + nv2
        four_pose_pairs.append((p1, p2))
        if draw_config:
            vis_3d(
                1500, 1500, o3d_pc(throw_outliers(pts)),
                draw_camera(full_K, p1, 1600, 1200),
                draw_camera(full_K, p2, 1600, 1200),
            )

    i = np.argmax(num_infront)
    t, R = four[i]
    p1, p2 = four_pose_pairs[i]
    return p1, p2, t, R


def F_from_K_and_E(K, E):
    return inv(K).T @ E @ inv(K)


def E_from_K_and_F(K, F):
    return K.T @ F @ K


def throw_outliers(pts):
    pts = pts[:, :3]
    mask = (np.abs(pts) > 10).any(axis=1)
    return pts[~mask]
