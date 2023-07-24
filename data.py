from PIL import Image
import matplotlib.pyplot as plt
import json

class ImageInfo():
    def __init__(self, images_dir, image_id):
        image_path = f"{images_dir}/{image_id}.jpg"
        self.image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.image.size
        intrinsics = json.load(open(f"{images_dir}/{image_id}_intrinsics.json"))
        self.horizontal_fov = intrinsics["horizontalFieldOfViewDegrees"]
    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.axis("off")  # remove the axis
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        plt.show()

ImageInfo('/Users/fak/work/Captures 3/Jul 23, 2023 at 8:59:51 PM', 'IMG_0001').show()
