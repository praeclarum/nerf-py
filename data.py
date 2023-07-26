import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import tqdm

import renderer


class ImageInfo:
    def __init__(self, images_dir, image_id, max_size, device):
        image_path = f"{images_dir}/{image_id}.jpg"
        self.image = Image.open(image_path).convert("RGB")
        if self.image.width > max_size or self.image.height > max_size:
            max_dim = max(self.image.width, self.image.height)
            scale = max_size / max_dim
            self.image = self.image.resize(
                (int(self.image.width * scale), int(self.image.height * scale)),
                resample=Image.BICUBIC,
            )
        self.width, self.height = self.image.size
        self.horizontal_fov_degrees = 62.3311
        self.intrinsics = torch.eye(4, device=device)
        intrinsics_txt_path = f"{images_dir}/{image_id}_Intrinsics.txt"
        intrinsics_json_path = f"{images_dir}/{image_id}_intrinsics.json"
        if os.path.exists(intrinsics_json_path):
            intrinsics = json.load(open(intrinsics_json_path))
            self.intrinsics = load_json_matrix(intrinsics, device)
            resolution = intrinsics["refSize"]
        elif os.path.exists(intrinsics_txt_path):
            self.intrinsics = load_matrix(intrinsics_txt_path, device)
            resolution = [
                int(float(x))
                for x in open(f"{images_dir}/{image_id}_Resolution.txt")
                .readlines()[0]
                .split()
            ]
        if self.image.width != resolution[0] or self.image.height != resolution[1]:
            # print(
            #     f"RESOLUTION MISMATCH {resolution} vs {self.image.width} {self.image.height}"
            # )
            scale = self.image.width / resolution[0]
            self.intrinsics *= scale
        self.cam_ray_dirs = renderer.get_intrinsic_cam_ray_dirs(
            self.image.width, self.image.height, self.intrinsics, device
        )
        self.set_extrinsics(torch.eye(4, device=device))
        extrinsics_txt_path = f"{images_dir}/{image_id}_Transform.txt"
        if os.path.exists(extrinsics_txt_path):
            self.set_extrinsics(load_matrix(extrinsics_txt_path, device))
        image_ar = np.array(self.image)
        self.image_tensor = torch.tensor(image_ar, device=device) / 255.0

    def set_extrinsics(self, extrinsics):
        self.extrinsics = extrinsics
        position = extrinsics[:3, 3]
        self.ray_origs = position.unsqueeze(0).unsqueeze(0).expand(
            self.height, self.width, 3
        )
        rot = extrinsics[:3, :3].unsqueeze(0).unsqueeze(0).expand(
            self.height, self.width, 3, 3
        )
        self.ray_dirs = torch.matmul(rot, self.cam_ray_dirs.unsqueeze(-1)).squeeze(-1)

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.axis("off")  # remove the axis
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        plt.show()
        plt.close()


def load_json_matrix(json_obj, device):
    col0 = torch.tensor(json_obj["matrixCol0"], device=device)
    col1 = torch.tensor(json_obj["matrixCol1"], device=device)
    col2 = torch.tensor(json_obj["matrixCol2"], device=device)
    if "matrixCol3" not in json_obj:
        col3 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    else:
        col3 = torch.tensor(json_obj["matrixCol3"], device=device)
    if col0.shape[0] == 3:
        col0 = torch.cat([col0, torch.tensor([0.0], device=device)])
    if col1.shape[0] == 3:
        col1 = torch.cat([col1, torch.tensor([0.0], device=device)])
    if col2.shape[0] == 3:
        col2 = torch.cat([col2, torch.tensor([0.0], device=device)])
    if col3.shape[0] == 3:
        col3 = torch.cat([col3, torch.tensor([1.0], device=device)])
    matrix = torch.stack([col0, col1, col2, col3], dim=1)
    return matrix


def load_matrix(path, device):
    lines = open(path).readlines()[:4]
    rows = [list(map(float, line.split()[1:])) for line in lines]
    matrix = torch.tensor(rows, device=device)
    return matrix


def load_images(images_dir, max_size, device):
    print(f"Loading images from {images_dir}")
    image_paths = sorted(glob.glob(f"{images_dir}/*.jpg"))
    images = []
    for image_path in tqdm.tqdm(image_paths):
        image_id = os.path.basename(image_path).replace(".jpg", "")
        image = ImageInfo(images_dir, image_id, max_size, device)
        images.append(image)
    poses_path = images_dir + "/poses.json"
    if os.path.exists(poses_path):
        poses = json.load(open(poses_path))
        if len(poses) != len(images):
            print(f"WARNING: len(poses) != len(images) {len(poses)} != {len(images)}")
        for i in range(len(images)):
            images[i].set_extrinsics(load_json_matrix(poses[i], device))
            # print(f"Extrinsics {i}:\n  {image.extrinsics.device}\n  {image.extrinsics}")
    return images


def get_cam_bounding_box(images):
    cam_positions = []
    for image in images:
        cam_pos = image.extrinsics[:3, 3]
        cam_positions.append(cam_pos)
    cam_positions = torch.stack(cam_positions, dim=0)
    cam_min = torch.min(cam_positions, dim=0)[0]
    cam_max = torch.max(cam_positions, dim=0)[0]
    return cam_min, cam_max


if __name__ == "__main__":
    images_dir = "/Volumes/home/Data/datasets/nerf/piano1"
    images = load_images(images_dir, 128, "cpu")
    get_cam_bounding_box(images)
    images[0].show()
    # ImageInfo('/home/fak/Data/datasets/nerf/eli2', 'Frame0_Image', 128).show()
