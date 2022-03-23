from pathlib import Path
import numpy as np


# # for mac os; somehow colmap import triggers error when using pudb
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    import pycolmap
    import pickle

    root = Path("~/work/colmap_data/0").expanduser()
    reconstruction = pycolmap.Reconstruction(root)
    print(reconstruction.summary())

    poses = []
    Ks = []
    fnames = []
    for image_id, image in reconstruction.images.items():
        fnames.append(image.name)

        po = np.identity(4)
        po[:3, :4] = image.inverse_projection_matrix()
        poses.append(po)

        cam = reconstruction.cameras[image.camera_id]
        intr = cam.calibration_matrix()
        Ks.append(intr)

    poses = np.array(poses)
    Ks = np.array(Ks)

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
        "pts_xyz": pts_xyz,
        "pts_rgb": pts_rgb
    }

    with open("recons.pkl", 'wb') as f:
        pickle.dump(recons, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
