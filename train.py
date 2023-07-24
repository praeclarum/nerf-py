import torch
import math
import data
import model
import renderer

nerf_model = model.DeepNeRF() 

render = renderer.ImageRenderer(
    nerf_model,
    70,
    50,
    math.radians(60.0),
    z_near=0.1,
    z_far=10.0,
    num_samples_per_ray=11,
)

camera_local_to_world = torch.eye(4)
renderer.show_image(render(camera_local_to_world))
