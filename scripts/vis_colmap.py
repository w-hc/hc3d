from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image

from fabric.io import load_object
from hc3d.vis import CameraCone
from hc3d.utils import batch_img_resize, inlier_mask


def main():
    img_root = Path("/scratch/colmap_data/south-building/images")
    recons = load_object("recons.pkl")

    print(recons.keys())

    pts = recons['pts_xyz']
    rgb = recons['pts_rgb'] / 255.
    mask = inlier_mask(pts)
    pts = pts[mask]
    rgb = rgb[mask]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(pts.astype(np.float32))
    pcd.point["colors"] = o3d.core.Tensor(rgb.astype(np.float32))

    fnames = recons['fnames']

    imgs = np.array([
        np.array(Image.open(img_root / fn)) for fn in fnames
    ])
    H, W = imgs.shape[1:3]
    imgs = batch_img_resize(imgs, 100)

    geoms = []
    for i, (po, K) in enumerate(zip(recons["poses"], recons["Ks"])):
        # # purposely set K wrong using negative-z convention; what happens to cones?
        # K = np.array([
        #     [K[0, 0], 0, -K[0, 2]],
        #     [0, -K[1, 1], -K[1, 2]],
        #     [0, 0, -1],
        # ])
        cne = CameraCone(K, po, W, H, top_left_corner=[0, 0], scale=0.2)
        lset = cne.as_line_set()
        view = cne.as_view_plane(imgs[i])
        geoms.extend([lset, view])

    o3d.visualization.draw([pcd, *geoms], show_skybox=False)


if __name__ == "__main__":
    main()
