import datetime
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


def checkpoint():
    global num_trained_steps
    out_path = f"{output_dir}/checkpoint_{num_trained_steps:04d}.pth"
    torch.save(
        {
            "num_trained_steps": num_trained_steps,
            "model_state_dict": nerf_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out_path,
    )


def render(cam_ray_dirs, cam_transform, num_samples_per_ray=12):
    return renderer.render(
        nerf_model,
        cam_ray_dirs,
        z_near=0.1,
        z_far=4.0,
        num_samples_per_ray=num_samples_per_ray,
        camera_local_to_world=cam_transform,
        include_view_direction=include_view_direction,
    )


def get_train_batch(crop_size):
    image = images[np.random.randint(0, len(images))]
    image_tensor = image.image_tensor
    height, width = image_tensor.shape[:2]
    if width > crop_size or height > crop_size:
        crop_y_index = np.random.randint(0, height - crop_size)
        crop_x_index = np.random.randint(0, width - crop_size)
        y = image_tensor[
            crop_y_index : crop_y_index + crop_size,
            crop_x_index : crop_x_index + crop_size,
        ]
        cam_ray_dirs = image.cam_ray_dirs[
            crop_y_index : crop_y_index + crop_size,
            crop_x_index : crop_x_index + crop_size,
        ]
    else:
        y = image_tensor
        cam_ray_dirs = image.cam_ray_dirs
    cam_transform = image.extrinsics
    return cam_ray_dirs, cam_transform, y


def sample(crop_size=384):
    global num_trained_steps
    samples = []
    nerf_model.eval()
    for sample_i in range(4):
        cam_ray_dirs, cam_transform, y = get_train_batch(crop_size=crop_size)
        depth_samples = [y.detach().cpu()]
        for depth in [0.0, -0.1, 0.1]:
            camera_local_to_world = torch.clone(cam_transform)
            camera_local_to_world[2, 3] += depth
            y_pred = render(cam_ray_dirs, camera_local_to_world, num_samples_per_ray=64).detach().cpu()
            depth_samples.append(y_pred)
        samples.append(torch.cat(depth_samples, dim=1))
    nerf_model.train()
    # renderer.show_image(y_pred)
    y_pred = torch.cat(samples, dim=0)
    out_path = f"{output_dir}/sample_{num_trained_steps:04d}.png"
    out_tmp_path = f"{tmp_dir}/sample_{num_trained_steps:04d}.png"
    Image.fromarray(np.uint8(y_pred.numpy() * 255)).save(out_tmp_path)
    os.rename(out_tmp_path, out_path)


def train_step(crop_size=32, num_accum=64):
    global num_trained_steps
    optimizer.zero_grad()
    total_loss = 0.0
    for i in range(num_accum):
        cam_ray_dirs, cam_transform, y = get_train_batch(crop_size=crop_size)
        y_pred = render(cam_ray_dirs, cam_transform)
        loss = torch.nn.functional.mse_loss(y_pred, y) / num_accum
        loss.backward()
        total_loss += loss.detach().cpu()
    optimizer.step()
    num_trained_steps += 1
    return total_loss


def train_loop(num_steps):
    p = tqdm.tqdm(range(num_steps))
    loss_sum = 0.0
    loss_count = 0
    for i in p:
        loss = train_step()
        loss_sum += loss
        loss_count += 1
        average_loss = loss_sum / loss_count
        p.set_description(f"loss={average_loss:.4f}")
    # checkpoint()
    sample()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device "{device}": {torch.cuda.get_device_name(device)}')

include_view_direction = False

# images_dir = "/home/fak/Data/datasets/nerf/desk1"
# image = data.ImageInfo(images_dir, "IMG_0001", 256)

dataset_name = "eli2"

images_dir = f"/Volumes/home/Data/datasets/nerf/{dataset_name}"
# image = data.ImageInfo(images_dir, "Frame0", 128, device)
# print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}")
# train_image = image.image_tensor.to(device)
images = data.load_images(images_dir, 256, device)

# nerf_model = model.DeepNeRF(include_view_direction=include_view_direction).to(device)
nerf_model = model.MildenhallNeRF(include_view_direction=include_view_direction, device=device).to(device)

num_trained_steps = 0
optimizer = torch.optim.Adam(
    nerf_model.parameters(),
    betas=(0.9, 0.99),
    eps=1e-15,
    lr=1e-2)

total_batch_size = 2**18 # = 262144

camera_local_to_world = torch.eye(4, device=device)

run_id = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

output_dir = f"/home/fak/nn/Data/generated/nerf/{dataset_name}/{run_id}"
os.makedirs(output_dir, exist_ok=True)
tmp_dir = f"/home/fak/nn/Data/generated/nerf/{dataset_name}/tmp"
os.makedirs(tmp_dir, exist_ok=True)
code_files = glob.glob(f"{os.path.dirname(__file__)}/*.py")
for code_file in code_files:
    shutil.copy(code_file, output_dir)

sample()

print(f"Training {run_id}...")
train_loop(2)
train_loop(64)
train_loop(128)
for i in range(32):
    train_loop(256)
