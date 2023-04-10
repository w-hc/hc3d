from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image

from fabric.io import load_object
from hc3d.vis import CameraCone
from hc3d.utils import batch_img_resize, inlier_mask

import click


@click.command()
@click.option(
    "-r", "--recons", help="the serialized reconstruction object", type=str
)
@click.option(
    "-i", "--img_root", help="image directory",
    default="/scratch/nerf_data/360_v2/garden/images_8"
)
def main(recons, img_root):
    img_root = Path(img_root)
    recons = load_object(recons)

    print(recons.keys())

    pts = recons['pts_xyz']
    rgb = recons['pts_rgb'] / 255.
    mask = inlier_mask(pts)
    pts = pts[mask]
    rgb = rgb[mask]

    pcd = o3d.t.geometry.PointCloud()
    # o3d is very finicky abt numeric type; little auto-conversion
    pcd.point["positions"] = o3d.core.Tensor(pts.astype(np.float32))
    pcd.point["colors"] = o3d.core.Tensor(rgb.astype(np.float32))

    fnames = recons['fnames']

    imgs = np.array([
        np.array(Image.open(img_root / fn)) for fn in fnames
    ])

    imgs = batch_img_resize(imgs, 100)

    geoms = []
    for i, (po, K, (H, W)) in enumerate(zip(
        recons["poses"], recons["Ks"], recons["HWs"]
    )):
        # # note: COLMAP uses OpenCV, right-handed, z-forward convention.
        # # my code here is agnostic to convention; things work out.
        # # But can purposely set K wrong using negative-z convention; what happens to cones?
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
