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


def sample():
    global num_trained_steps
    samples = []
    render.eval()
    for depth in [-1.0, 0.0, 1.0]:
        depth_samples = []
        for angle in [15, 0, -15]:
            y_rot_rads = math.radians(angle)
            camera_local_to_world = torch.eye(4, device=device)
            camera_local_to_world[0, 0] = math.cos(y_rot_rads)
            camera_local_to_world[0, 2] = math.sin(y_rot_rads)
            camera_local_to_world[2, 0] = -math.sin(y_rot_rads)
            camera_local_to_world[2, 2] = math.cos(y_rot_rads)
            camera_local_to_world[2, 3] = depth
            y_pred = render(camera_local_to_world).detach().cpu()
            depth_samples.append(y_pred)
        samples.append(torch.cat(depth_samples, dim=1))
    render.train()
    # renderer.show_image(y_pred)
    y_pred = torch.cat(samples, dim=0)
    out_path = f"{output_dir}/sample_{num_trained_steps:04d}.png"
    Image.fromarray(np.uint8(y_pred.numpy() * 255)).save(out_path)


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
    checkpoint()
    sample()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device "{device}": {torch.cuda.get_device_name(device)}')

include_view_direction = False

# images_dir = "/Users/fak/work/Captures 3/Jul 23, 2023 at 8:59:51 PM"
images_dir = "/home/fak/Data/datasets/nerf/desk1"

image = data.ImageInfo(images_dir, "IMG_0001", 256)
# image.show()
print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}, HFOV {image.horizontal_fov}")
train_image = image.image_tensor.to(device)

nerf_model = model.DeepNeRF(include_view_direction=include_view_direction).to(device)

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
    include_view_direction=include_view_direction,
).to(device)

camera_local_to_world = torch.eye(4, device=device)

run_id = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

output_dir = f"/home/fak/nn/Data/generated/nerf/desk1/{run_id}"
os.makedirs(output_dir, exist_ok=True)
code_files = glob.glob(f"{os.path.dirname(__file__)}/*.py")
for code_file in code_files:
    shutil.copy(code_file, output_dir)

sample()

train_loop(2)
train_loop(64)
train_loop(256)
train_loop(1024)
for i in range(16):
    train_loop(2048)
