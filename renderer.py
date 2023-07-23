import math
import torch
import matplotlib.pyplot as plt


#
# Renderer takes a radiance function:
#  radiance(position, view_direction) -> (density, color (rgb))
#  where density is a scalar and color is a 3-vector
#  position and view_direction are 3-vectors
#
# Width, height image output size
# vertical_fov is the vertical field of view in radians
# z_near and z_far are the near and far clipping planes
# 
# Position and orientation of the camera represented as a 4x4 matrix
#

#
# ALGORITHM
#
# For each pixel in the output image:
#   Compute the ray origin (same as camera position)
#   Compute the ray direction
#   Create bins along the ray
#   For each bin, create a random sample point
#   For each sample point, compute the radiance and density
#   Accumulate the radiance and density to get final output color
#

def render(radiance,
           width, height, vertical_fov,
           z_near, z_far, num_samples_per_ray,
           camera_local_to_world):
    # ray_origins are (1, 3)
    ray_origins = camera_local_to_world[:3, 3].view(1, 3)
    # print("ray_origins", ray_origins.shape)
    # ray_dirs are (H, W, 3)
    ray_dirs = get_ray_dirs(width, height, vertical_fov, camera_local_to_world)
    # print("ray_dirs", ray_dirs.shape)

    # sample_distances are (H * W, num_samples_per_ray)
    sample_distances = \
        torch.rand((height * width, num_samples_per_ray)) * (z_far - z_near) + z_near
    # print("sample_distances.shape", sample_distances.shape)
    # print("sample_distances", sample_distances.shape)
    
    # sample_positions are (H * W, num_samples_per_ray, 3)
    sample_positions = ray_origins + \
        ray_dirs.view(height * width, 1, 3) * \
        sample_distances.view(height * width, num_samples_per_ray, 1)
    # print("sample_positions.shape", sample_positions.shape)
    # print("sample_positions", sample_positions)

    # sample_dirs are (H * W, num_samples_per_ray, 3)
    sample_dirs = ray_dirs.view(height * width, 1, 3).repeat(1, num_samples_per_ray, 1)
    # print("sample_dirs", sample_dirs.shape)

    # set sample_radiance shape to (H * W * num_samples_per_ray, 3) for radiance function
    sample_densities, sample_colors = radiance(
        sample_positions.view(height * width * num_samples_per_ray, 3),
        sample_dirs.view(height * width * num_samples_per_ray, 3))
    
    sample_densities = sample_densities.view(height * width, num_samples_per_ray)
    sample_colors = sample_colors.view(height * width, num_samples_per_ray, 3)
    # print("sample_densities.shape", sample_densities.shape)
    # print("sample_densities", sample_densities)
    # print("sample_colors", sample_colors.shape)
    sample_total_densities = torch.sum(sample_densities, dim=1)
    # print("sample_total_densities", sample_total_densities.shape)
    sample_weighted_colors = sample_colors * sample_densities.view(height * width, num_samples_per_ray, 1)
    # print("sample_weighted_colors", sample_weighted_colors.shape)
    sample_total_colors = torch.sum(sample_weighted_colors, dim=1)
    # print("sample_total_colors.shape", sample_total_colors.shape)
    # print("sample_total_colors", sample_total_colors)
    pixel_colors = sample_total_colors / sample_total_densities.view(height * width, 1)
    # print("pixel_colors", pixel_colors.shape)
    pixel_colors = pixel_colors.view(height, width, 3)
    # print("pixel_colors", pixel_colors.shape)

    return pixel_colors

def get_ray_dirs(width, height,
                 vertical_fov_radians,
                 camera_local_to_world):
    # compute the camera horizontal field of view in radians
    horizontal_fov_radians = vertical_fov_radians * width / height

    # compute ray directions for every pixel in camera space
    rot_y_rads = torch.linspace(-horizontal_fov_radians / 2.0,
                                horizontal_fov_radians / 2.0,
                                width)
    rot_x_rads = torch.linspace(vertical_fov_radians / 2.0,
                                -vertical_fov_radians / 2.0,
                                height)
    rot_y_rads = rot_y_rads.view(1, width).repeat(height, 1)
    rot_x_rads = rot_x_rads.view(height, 1).repeat(1, width)
    cam_ray_x_dir = -torch.sin(rot_y_rads) * torch.cos(rot_x_rads)
    cam_ray_y_dir = torch.sin(rot_x_rads)
    cam_ray_z_dir = -torch.cos(rot_y_rads) * torch.cos(rot_x_rads)
    cam_ray_dir = torch.stack([cam_ray_x_dir, cam_ray_y_dir, cam_ray_z_dir], dim=2)

    # Get the transform that rotates local camera space vectors into
    # world space vector
    camera_local_to_world_rot = camera_local_to_world[:3, :3]
    ray_dir = torch.matmul(
        camera_local_to_world_rot,
        cam_ray_dir.view(height * width, 3, 1))
    ray_dir = ray_dir.view(height, width, 3)
    return ray_dir

def sphere_radiance(position, view_direction):
    # position is (N, 3)
    # view_direction is (N, 3)
    # returns (N, 1) density and (N, 3) color
    center = torch.tensor([0.0, 0.0, -5.0])
    radius = 1.0
    distance_to_surface = torch.norm(position - center.view(1, 3), dim=1, keepdim=True) - radius
    density = torch.clamp(torch.relu(-distance_to_surface), 0.0, 1.0)
    color = torch.tensor([0.2, 1.0, 0.0]).view(1, 3).repeat(position.shape[0], 1)
    return (density, color)

def show_image(image):
    plt.imshow(image)
    plt.show()

y_rot_rads = math.radians(0.0)
camera_local_to_world=torch.eye(4)
camera_local_to_world[0, 0] = math.cos(y_rot_rads)
camera_local_to_world[0, 2] = math.sin(y_rot_rads)
camera_local_to_world[2, 0] = -math.sin(y_rot_rads)
camera_local_to_world[2, 2] = math.cos(y_rot_rads)

show_image(
    render(
        sphere_radiance,
        700, 500,
        math.radians(60.0),
        z_near=3.0, z_far=7.0,
        num_samples_per_ray=20,
        camera_local_to_world=camera_local_to_world))

