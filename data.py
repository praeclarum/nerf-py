import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

class ImageInfo():
    def __init__(self, images_dir, image_id, max_size):
        image_path = f"{images_dir}/{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        max_dim = max(width, height)
        scale = max_size / max_dim
        self.image = image.resize((int(width * scale), int(height * scale)))
        self.width, self.height = self.image.size
        image_ar = np.array(self.image)
        self.image_tensor = torch.tensor(image_ar) / 255.0
        intrinsics = json.load(open(f"{images_dir}/{image_id}_intrinsics.json"))
        self.horizontal_fov = intrinsics["horizontalFieldOfViewDegrees"][0]
    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.axis("off")  # remove the axis
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        plt.show()

# ImageInfo('/Users/fak/work/Captures 3/Jul 23, 2023 at 8:59:51 PM', 'IMG_0001').show()
