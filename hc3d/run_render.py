from pathlib import Path
import numpy as np
from numpy.linalg import inv
import open3d as o3d
import matplotlib.pyplot as plt

from .vis import OpenVisWrapper, generate_spiral_camera_trajectory, draw_camera
from .render import (
    compute_intrinsics,
    compute_extrinsics,
    simple_point_render
)

root = Path("./data")


def four_debug_points():
    z = -1
    pts = np.array([
        [0, 0, z],
        [1, -1, z],
        [-1, -1, z],
        [0, 1, z],
    ])
    return pts


def load_horse_mesh():
    fname = str(root / "horse.obj")
    mesh = o3d.io.read_triangle_mesh(fname, True)
    return mesh


def load_horse_pointcloud():
    fname = str(root / "horse.obj")
    mesh = o3d.io.read_triangle_mesh(fname, True)
    # sample a point cloud from the surface of the mesh
    pcloud = mesh.sample_points_uniformly(10000)
    points = np.asarray(pcloud.points)
    return points


def load_rockfeller_facade():
    fname = str(root / "rockfeller_chapel.ply")
    pcd = o3d.io.read_point_cloud(fname)
    return pcd


def debug_four_points():
    z = -1
    pts = np.array([
        [0, 0, z],
        [1, -1, z],
        [-1, -0.5, z],
        [0, 1, z],
    ])

    eye = np.array([0, 0, 2])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])

    img = simple_point_render(pts, 80, 60, 90, eye, front, up)
    plt.imshow(img)
    plt.show()


def mine_vs_open3d_pc():
    pts = load_horse_pointcloud()

    img_w, img_h = 600, 450
    fov = 90

    # view 1
    eye = np.array([0, 0, 2])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])

    # view 2
    # eye = np.array([2, 0, 0.5])
    # front = np.array([-1, 0, 0])
    # up = np.array([0, 1, 0])

    mine = simple_point_render(pts, img_w, img_h, fov, eye, front, up)

    mesh = load_horse_mesh()
    renderer = OpenVisWrapper(img_w, img_h)
    renderer.add(mesh)

    K = compute_intrinsics(img_w / img_h, fov, img_h)
    E = compute_extrinsics(eye, front, up)
    renderer.set_intrinsics(K)
    renderer.set_extrinsics(E)
    ref = renderer.to_image(None)

    plt.imshow(ref)
    plt.imshow(mine, alpha=0.5)
    plt.show()


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
    renderer.set_intrinsics(K)
    renderer.set_extrinsics(E)
    renderer.blocking_run()


def vis_rockfeller():
    img_w, img_h = 1800, 1200
    fov = 60
    eye = np.array([0, 0, 2])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])

    pcd = load_rockfeller_facade()
    xyz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

    renderer = OpenVisWrapper(img_w, img_h)
    renderer.add(xyz)
    renderer.add(pcd)
    renderer.opts.mesh_show_wireframe = False

    eye = np.array([0, 0, 4])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, 0])

    extr = compute_extrinsics(eye, front, up)
    intr = compute_intrinsics(img_w / img_h, fov, img_h)
    renderer.set_intrinsics(intr)
    renderer.set_extrinsics(extr)

    renderer.vis.update_renderer()
    renderer.blocking_run()
    renderer.vis.destroy_window()


def vis_camera_trajectory():
    img_w, img_h = 1600, 1200
    fov = 60
    eye = np.array([0, 0, 3])
    front = np.array([0, 0, -1])
    up = np.array([0, 1, -1])

    mesh = load_horse_mesh()
    renderer = OpenVisWrapper(img_w, img_h)
    renderer.add(mesh)

    K = compute_intrinsics(img_w / img_h, fov, img_h)
    main_E = compute_extrinsics(eye, front, up)

    # first draw the main camera
    cam = draw_camera(K, inv(main_E), img_w, img_h)
    renderer.add(cam)

    # then draw the cameras on the spiral trajectory
    # you can use this to generate animation; we won't do this in this assignment
    # it's just busy work; leave it for the project
    traj_Es = generate_spiral_camera_trajectory(mesh.vertices, num_rounds=1)
    for E in traj_Es:
        cam = draw_camera(K, E, img_w, img_h)
        renderer.add(cam)

    renderer.set_intrinsics(K)
    renderer.set_extrinsics(main_E)

    renderer.blocking_run()


def main():
    pass

    # uncomment the function for the problem you are running.
    # run them one at a time.

    # # problem 3
    # debug_four_points()

    # # problem 4
    # mine_vs_open3d_pc()

    # # problem 5
    # vis_rockfeller()

    # # problem 6
    # vis_horse(fov=160)

    # # problem 7
    vis_camera_trajectory()


if __name__ == "__main__":
    main()
