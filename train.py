import torch
import numpy as np
import math
import data
import model
import renderer
import tqdm

image = data.ImageInfo(
    "/Users/fak/work/Captures 3/Jul 23, 2023 at 8:59:51 PM", "IMG_0001", 256
)
image.show()
print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}, HFOV {image.horizontal_fov}")

nerf_model = model.DeepNeRF()

render = renderer.ImageRenderer(
    nerf_model,
    image.width,
    image.height,
    horizontal_fov=math.radians(image.horizontal_fov),
    z_near=0.1,
    z_far=5.0,
    num_samples_per_ray=11,
)


camera_local_to_world = torch.eye(4)


def sample():
    render.eval()
    y_pred = render(camera_local_to_world)
    renderer.show_image(y_pred)
    render.train()


sample()

optimizer = torch.optim.Adam(nerf_model.parameters(), lr=1e-2)


def train_step():
    optimizer.zero_grad()
    y = image.image_tensor
    y_pred = render(camera_local_to_world)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach().tolist()


def train_loop(num_steps=64):
    p = tqdm.tqdm(range(num_steps))
    for i in p:
        loss = train_step()
        p.set_description(f"loss={loss:.4f}")
    sample()


train_loop(1)
train_loop(2)
train_loop(4)
train_loop(8)
train_loop(16)
train_loop(32)
train_loop()
train_loop()
train_loop()
train_loop()
train_loop()
train_loop()
train_loop()
train_loop()
