from collections import defaultdict
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import torch
import fire


def load_single_traj_poses(cam_traj_dir):
    with h5py.File(cam_traj_dir / "camera_keyframe_frame_indices.hdf5", "r") as f:
        inds = f['dataset']
        n = len(inds)
        # I don't really know what the inds are for; but just confirm no surprises
        assert (inds == np.arange(n)).all()

    with h5py.File(cam_traj_dir / "camera_keyframe_look_at_positions.hdf5", "r") as f:
        lookat = np.array(f['dataset']).astype(np.float32)
        assert lookat.shape == (n, 3)

    with h5py.File(cam_traj_dir / "camera_keyframe_orientations.hdf5", "r") as f:
        Rs = np.array(f['dataset']).astype(np.float32)
        assert Rs.shape == (n, 3, 3)

    with h5py.File(cam_traj_dir / "camera_keyframe_positions.hdf5", "r") as f:
        pos = np.array(f['dataset']).astype(np.float32)
        assert pos.shape == (n, 3)

    front = lookat - pos
    front = front / np.linalg.norm(front, axis=1, keepdims=True)
    neg_zs = -Rs[:, :, -1]  # -ve z is front
    # see https://github.com/apple/ml-hypersim/issues/67
    # the 0-th front is not expected to be aligned
    if not np.allclose(front[1:], neg_zs[1:], atol=1e-5):
        print(f"warning: {cam_traj_dir} lookat and R mismatch")

    Ts = pos

    # Rs = einops.rearrange(Rs, "n h w -> n w h", h=3, w=3)
    # x-axis points right, the positive y-axis points up,
    # and the positive z-axis points away from where the camera is looking
    # sounds like OpenGL cam convention
    poses = np.concatenate([Rs, Ts[..., None]], axis=-1)
    # pad so that poses [n, 3, 4] -> [n, 4, 4] with 0 filling on new entries
    poses = np.pad(poses, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    poses[:, 3, 3] = 1

    poses = poses.astype(np.float32)
    return torch.as_tensor(poses)


def parse_cam_traj_csv(fname):
    rows = []
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) == 1  # each row only has 1 comma separated item
            rows.append(row[0])

    rows = rows[1:]  # 1st row is header
    return rows


def load_all_poses():
    # root = Path("/scratch/omni3d_data/ml-hypersim/hypersim/ai_001_008")
    root = Path("/share/data/2pals/hypersim")
    all_scenes = list(sorted(root.iterdir()))

    all_poses = defaultdict(dict)

    for scene in tqdm(all_scenes):
        assert scene.is_dir()
        cam_traj_csv = scene / "_detail" / "metadata_cameras.csv"
        trajs = parse_cam_traj_csv(cam_traj_csv)

        # make sure the naming convention is [cam_00, cam_01, ... cam_99]
        # assert trajs == [f"cam_{i:0>2}" for i in range(len(trajs))], f"{scene.name}: {trajs}"
        # sadly this is not true, e.g. ai_007_004 has [cam_00, cam_02, cam_03]

        for tj in trajs:
            cam_traj_dir = scene / "_detail" / tj
            if not cam_traj_dir.is_dir():
                print(f"warning: {cam_traj_dir} doesn't exist")
                continue
            poses = load_single_traj_poses(cam_traj_dir)
            all_poses[scene.name][tj] = poses

    # convert to regular dict
    all_poses = dict(all_poses)

    torch.save(all_poses, "/scratch/whc/hypersim_poses.pt")


def check_pose_image_match():
    img_root = Path("/scratch/omni3d_data/ml-hypersim/hypersim")
    all_poses = torch.load("./hypersim_poses.pt")
    scenes = list(sorted(all_poses.keys()))

    for sn in scenes:
        trajs = list(sorted(all_poses[sn].keys()))
        for tj in trajs:
            poses = all_poses[sn][tj]
            n = len(poses)
            im_dir = img_root / sn / "images" / f"scene_{tj}_final_preview"
            assert im_dir.is_dir()
            im_fnames = list(sorted(im_dir.iterdir()))
            if len(im_fnames) != n:
                print(f"{sn}/{tj}: {len(im_fnames)} imgs vs {n} poses")

    pass


if __name__ == "__main__":
    fire.Fire()
