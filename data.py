import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

import renderer


class ImageInfo:
    def __init__(self, images_dir, image_id, max_size, device):
        image_path = f"{images_dir}/{image_id}_Image.jpg"
        self.image = Image.open(image_path).convert("RGB")
        self.horizontal_fov_degrees = 62.3311
        self.intrinsics = torch.eye(4)
        self.extrinsics = torch.eye(4)
        intrinsics_txt_path = f"{images_dir}/{image_id}_Intrinsics.txt"
        extrinsics_txt_path = f"{images_dir}/{image_id}_Transform.txt"
        intrinsics_json_path = f"{images_dir}/{image_id}_intrinsics.json"
        if os.path.exists(intrinsics_json_path):
            intrinsics = json.load(open(intrinsics_json_path))
            self.horizontal_fov_degrees = intrinsics["horizontalFieldOfViewDegrees"][0]
            self.cam_ray_dirs = renderer.get_cam_ray_dirs(
                self.width, self.height, self.horizontal_fov_degrees, device
            )
        elif os.path.exists(intrinsics_txt_path):
            self.intrinsics = load_matrix(intrinsics_txt_path, device)
            resolution = [
                int(float(x))
                for x in open(f"{images_dir}/{image_id}_Resolution.txt")
                .readlines()[0]
                .split()
            ]
            if self.image.width != resolution[0] or self.image.height != resolution[1]:
                print(
                    f"RESOLUTION MISMATCH {resolution} vs {self.image.width} {self.image.height}"
                )
                self.image = self.image.resize((resolution[0], resolution[1]))
            self.cam_ray_dirs = renderer.get_intrinsic_cam_ray_dirs(
                self.image.width, self.image.height, self.intrinsics, device
            )
        if os.path.exists(extrinsics_txt_path):
            self.extrinsics = load_matrix(extrinsics_txt_path, device)
        # print(f"Intrinsics:\n{self.intrinsics}")
        # print(f"Extrinsics:\n{self.extrinsics}")
        self.width, self.height = self.image.size
        image_ar = np.array(self.image)
        self.image_tensor = torch.tensor(image_ar) / 255.0

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.axis("off")  # remove the axis
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        plt.show()
        plt.close()


def load_matrix(path, device):
    lines = open(path).readlines()[:4]
    rows = [list(map(float, line.split()[1:])) for line in lines]
    matrix = torch.tensor(rows, device=device)
    return matrix


if __name__ == "__main__":
    images_dir = "/Volumes/home/Data/datasets/nerf/eli2"
    ImageInfo(images_dir, "Frame0", 128, "cpu").show()
    # ImageInfo('/home/fak/Data/datasets/nerf/eli2', 'Frame0_Image', 128).show()
