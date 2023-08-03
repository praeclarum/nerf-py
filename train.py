import sys
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


def load_checkpoint(path):
    global num_trained_steps, run_id
    checkpoint = torch.load(path)
    num_trained_steps = checkpoint["num_trained_steps"]
    nerf_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    run_id = os.path.basename(os.path.dirname(path))


def render_image(cam_ray_dirs, cam_transform, num_samples_per_ray=64):
    return renderer.render_image(
        nerf_model,
        cam_ray_dirs,
        z_near=0.1,
        z_far=4.0,
        num_samples_per_ray=num_samples_per_ray,
        camera_local_to_world=cam_transform,
        include_view_direction=include_view_direction,
    )


def render_rays(ray_origs, ray_dirs, num_samples_per_ray=16):
    return renderer.render_rays(
        nerf_model,
        ray_origs=ray_origs,
        ray_dirs=ray_dirs,
        z_near=0.1,
        z_far=2.0,
        num_samples_per_ray=num_samples_per_ray,
        include_view_direction=include_view_direction,
    )


def sample(crop_size=384):
    global num_trained_steps
    samples = []
    nerf_model.eval()
    for sample_i in range(4):
        cam_ray_dirs, cam_transform, y = get_train_image(crop_size=crop_size)
        depth_samples = [y.detach().cpu()]
        for depth in [0.0, -0.1, 0.1]:
            camera_local_to_world = torch.clone(cam_transform)
            camera_local_to_world[2, 3] += depth
            y_pred = (
                render_image(
                    cam_ray_dirs, camera_local_to_world, num_samples_per_ray=64
                )
                .detach()
                .cpu()
            )
            depth_samples.append(y_pred)
        samples.append(torch.cat(depth_samples, dim=1))
    nerf_model.train()
    # renderer.show_image(y_pred)
    y_pred = torch.cat(samples, dim=0)
    out_path = f"{output_dir}/sample_{num_trained_steps:04d}.png"
    out_tmp_path = f"{tmp_dir}/sample_{num_trained_steps:04d}.png"
    Image.fromarray(np.uint8(y_pred.numpy() * 255)).save(out_tmp_path)
    os.rename(out_tmp_path, out_path)


def get_train_image(crop_size):
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


def get_train_rays(batch_size):
    num_images = image_color.shape[0]
    height, width, _ = image_color[0].shape
    batch_image_index = np.random.randint(0, num_images, size=batch_size)
    batch_y = np.random.randint(0, height, size=batch_size)
    batch_x = np.random.randint(0, width, size=batch_size)
    batch_color = image_color[batch_image_index, batch_y, batch_x]
    batch_ray_dir = image_ray_dir[batch_image_index, batch_y, batch_x]
    batch_ray_orig = image_ray_orig[batch_image_index, batch_y, batch_x]
    return batch_ray_orig, batch_ray_dir, batch_color


def train_step(batch_size=2**16):
    global num_trained_steps
    optimizer.zero_grad()
    ray_origs, ray_dirs, ray_colors = get_train_rays(batch_size=batch_size)
    ray_colors_pred = render_rays(ray_origs, ray_dirs)
    loss = torch.nn.functional.mse_loss(ray_colors_pred, ray_colors)
    loss.backward()
    detached_loss = loss.detach().cpu()
    optimizer.step()
    num_trained_steps += 1
    return detached_loss


def train_loop(num_steps):
    global last_auto_checkpoint_time
    p = tqdm.tqdm(range(num_steps))
    loss_sum = 0.0
    loss_count = 0
    for i in p:
        loss = train_step()
        loss_sum += loss
        loss_count += 1
        average_loss = loss_sum / loss_count
        p.set_description(f"loss={average_loss:.4f}")
    if (
        last_auto_checkpoint_time is None
        or (datetime.datetime.now() - last_auto_checkpoint_time).total_seconds()
        > 60 * 5
    ):
        checkpoint()
        last_auto_checkpoint_time = datetime.datetime.now()
    sample()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
print(f'Using device "{device}": {device_name}')

include_view_direction = True

dataset_name = sys.argv[1]

images_dir = f"/Volumes/home/Data/datasets/nerf/{dataset_name}"
# image = data.ImageInfo(images_dir, "Frame0", 128, device)
# print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}")
# train_image = image.image_tensor.to(device)
images = data.load_images(images_dir, 128, device)
image_color = torch.cat([image.image_tensor.unsqueeze(0) for image in images], dim=0)
image_ray_dir = torch.cat([image.ray_dirs.unsqueeze(0) for image in images], dim=0)
image_ray_orig = torch.cat([image.ray_origs.unsqueeze(0) for image in images], dim=0)

# nerf_model = model.DeepNeRF(include_view_direction=include_view_direction).to(device)
nerf_model = model.MildenhallNeRF(
    include_view_direction=include_view_direction, device=device
).to(device)

num_trained_steps = 0
last_auto_checkpoint_time = None
optimizer = torch.optim.Adam(
    nerf_model.parameters(), betas=(0.9, 0.99), eps=1e-15, lr=1e-2
)

camera_local_to_world = torch.eye(4, device=device)

run_id = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

# load_checkpoint("/Volumes/nn/Data/generated/nerf/piano3/2023-08-02_16-11-02/checkpoint_35716.pth")

output_dir = f"/home/fak/nn/Data/generated/nerf/{dataset_name}/{run_id}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving to {output_dir}...")
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
train_loop(256)
train_loop(512)
checkpoint()
for i in range(32):
    train_loop(1024)
checkpoint()
