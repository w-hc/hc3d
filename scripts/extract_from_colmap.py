"""
This script is used to extract info from colmap database. It uses
pycolmap, https://github.com/colmap/pycolmap, a library developed by students in
Pollefeys' group.

The library as of now (2023.4.10) doesn't have documentation. You can guess its
behavior from the names of python bindings, such as this
https://github.com/colmap/pycolmap/blob/401f82658cdad1e8b657c77381863f9e261c7c3c/reconstruction/camera.cc#L107
"""

from pathlib import Path
import pickle
import numpy as np
import pycolmap

# # for mac os; somehow colmap import triggers error when using pudb
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import click


@click.command()
@click.option(
    '-r', '--root', default="/scratch/nerf_data/360_v2/garden/sparse/0",
    help='dir where the colmap database .bin files are stored'
)
@click.option(
    '-o', '--ofname', default="recons.pkl", help='output pickle file name'
)
def main(root, ofname):
    root = Path(root)

    reconstruction = pycolmap.Reconstruction(root)
    print(reconstruction.summary())

    # reconstruction.export_CAM("./")  alternatively consider this for camera exporting

    poses = []
    Ks = []
    fnames = []
    HWs = []
    for image_id, image in reconstruction.images.items():
        fnames.append(image.name)

        po = np.identity(4)
        po[:3, :4] = image.inverse_projection_matrix()
        poses.append(po)

        cam = reconstruction.cameras[image.camera_id]
        HWs.append([cam.height, cam.width])
        # WARN: this is disregarding the camera model, pinhole/fisheye, etc.
        intr = cam.calibration_matrix()
        Ks.append(intr)

    poses = np.array(poses)
    Ks = np.array(Ks)
    HWs = np.array(HWs)

    pts_xyz = []
    pts_rgb = []
    for point3D_id, point3D in reconstruction.points3D.items():
        pts_xyz.append(point3D.xyz)
        pts_rgb.append(point3D.color)

    pts_xyz = np.array(pts_xyz)
    pts_rgb = np.array(pts_rgb, dtype=np.uint8)

    for camera_id, camera in reconstruction.cameras.items():
        print(camera_id, camera)

    recons = {
        "fnames": fnames,
        "poses": poses,
        "Ks": Ks,
        "HWs": HWs,
        "pts_xyz": pts_xyz,
        "pts_rgb": pts_rgb
    }  # could have dumped it as .json

    ofname = Path(ofname).with_suffix(".pkl")
    with open(ofname, 'wb') as f:
        pickle.dump(recons, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
