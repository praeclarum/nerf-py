import os
import shutil
import glob
import torch
import numpy as np
import math
import data
import model
import renderer
import tqdm
from PIL import Image


def sample():
    global num_trained_steps
    render.eval()
    y_pred = render(camera_local_to_world).detach().cpu()
    render.train()
    renderer.show_image(y_pred)
    out_path = f"{output_dir}/sample_{num_trained_steps:04d}.png"
    Image.fromarray(np.uint8(y_pred.numpy() * 255)).save(out_path)


def train_step():
    global num_trained_steps
    optimizer.zero_grad()
    y = train_image
    y_pred = render(camera_local_to_world)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    num_trained_steps += 1
    return loss.detach().cpu().tolist()


def train_loop(num_steps):
    p = tqdm.tqdm(range(num_steps))
    for i in p:
        loss = train_step()
        p.set_description(f"loss={loss:.4f}")
    sample()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device "{device}": {torch.cuda.get_device_name(device)}')

# images_dir = "/Users/fak/work/Captures 3/Jul 23, 2023 at 8:59:51 PM"
images_dir = "/home/fak/Data/datasets/nerf/desk1"

import datetime
run_id = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

output_dir = f"/home/fak/nn/Data/generated/nerf/desk1/{run_id}"
os.makedirs(output_dir, exist_ok=True)
code_files = glob.glob(f"{os.path.dirname(__file__)}/*.py")
for code_file in code_files:
    shutil.copy(code_file, output_dir)

image = data.ImageInfo(images_dir, "IMG_0001", 256)
image.show()
print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}, HFOV {image.horizontal_fov}")
train_image = image.image_tensor.to(device)

nerf_model = model.DeepNeRF().to(device)

num_trained_steps = 0
optimizer = torch.optim.Adam(nerf_model.parameters(), lr=1e-2)

render = renderer.ImageRenderer(
    nerf_model,
    image.width,
    image.height,
    horizontal_fov=math.radians(image.horizontal_fov),
    z_near=0.1,
    z_far=5.0,
    num_samples_per_ray=11,
    device=device,
).to(device)

camera_local_to_world = torch.eye(4, device=device)

sample()

train_loop(1)
train_loop(16)
train_loop(32)
for i in range(16):
    train_loop(512)
