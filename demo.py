import numpy as np
from hc3d.vis import draw_camera, quick_vis_3d, generate_spiral_camera_trajectory
from hc3d.render import compute_intrinsics


def draw_cam_cones():
    pts3d = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    poses = generate_spiral_camera_trajectory(pts3d)

    H, W = 500, 500
    fov = 60

    K = compute_intrinsics(W / H, fov, H)
    cones = []
    for po in poses:
        cone = draw_camera(K, po, H, W, scale=0.2)
        cones.append(cone)

    quick_vis_3d(*cones)


def main():
    draw_cam_cones()


if __name__ == "__main__":
    main()
