import math
import torch
import matplotlib.pyplot as plt

def render(radiance,
           width, height, vertical_fov,
           z_near, z_far, num_samples_per_ray,
           camera_local_to_world):
    """
    Renderer takes a radiance function:

    radiance(position, view_direction) -> (density, color (rgb))
    
    where density is a scalar and color is a 3-vector
    position and view_direction are 3-vectors

    Width, height image output size
    vertical_fov is the vertical field of view in radians
    z_near and z_far are the near and far clipping planes

    Position and orientation of the camera represented as a 4x4 matrix
    """
    # Determine the ray origin and direction for each pixel
    num_rays = width * height
    ray_origins = camera_local_to_world[:3, 3].view(1, 3)
    cam_ray_dirs = get_cam_ray_dirs(width, height, vertical_fov)
    ray_dirs = get_ray_dirs(cam_ray_dirs, camera_local_to_world)

    # Get the sample points along each ray
    sample_distances = sample_binned_uniform_distances(z_near, z_far, num_rays, num_samples_per_ray)
    sample_positions = ray_origins + \
        ray_dirs.view(num_rays, 1, 3) * \
        sample_distances.view(num_rays, num_samples_per_ray, 1)
    sample_dirs = ray_dirs.view(num_rays, 1, 3).repeat(1, num_samples_per_ray, 1)

    # Get the density and color at each sample point
    sample_densities, sample_colors = radiance(
        sample_positions.view(num_rays * num_samples_per_ray, 3),
        sample_dirs.view(num_rays * num_samples_per_ray, 3))
    sample_densities = sample_densities.view(num_rays, num_samples_per_ray)
    sample_colors = sample_colors.view(num_rays, num_samples_per_ray, 3)
    
    # Integrate the density and color along the ray
    mean_sample_separation = (z_far - z_near) / num_samples_per_ray
    colors = alpha_composite(sample_densities, sample_colors, mean_sample_separation, sample_distances.view(num_rays, num_samples_per_ray))
    colors = colors.view(height, width, 4)
    return colors

def alpha_composite(sample_densities, sample_colors, mean_sample_separation, sample_distances):
    """
    Composite colors using alpha blending

    sample_density shape (num_rays, num_samples_per_ray)
    sample_colors shape (num_rays, num_samples_per_ray, 3)
    mean_sample_separation is a scalar
    sample_distances shape (num_rays, num_samples_per_ray)
    """
    num_rays, num_samples_per_ray = sample_densities.shape
    sample_separations = sample_distances[:, 1:] - sample_distances[:, :-1]
    scaled_densities = sample_densities[:, :-1] * sample_separations
    alphas = 1.0 - torch.exp(-scaled_densities)
    # print(alphas[700*500*11//2])
    composite_alpha = 1.0 - torch.exp(-sample_densities[:, -1:] * mean_sample_separation)
    composite_color = sample_colors[:, -1]
    for i in range(num_samples_per_ray - 2, -1, -1):
        alpha_i = torch.unsqueeze(alphas[:, i], 1)
        one_minus_alpha_i = 1.0 - alpha_i
        # raise Exception(f"alpha_i.shape = {alpha_i.shape}")
        composite_color = alpha_i * sample_colors[:, i, :] + one_minus_alpha_i * composite_color
        # raise Exception(f"composite_color.shape = {composite_color.shape}")
        composite_alpha = alpha_i + one_minus_alpha_i * composite_alpha
        # print(composite_alpha[700*500//2+700])
        # raise Exception(f"composite_alpha.shape = {composite_alpha.shape}")
    # raise Exception(f"sample_distances.shape = {sample_distances.shape}")
    composite_rgba = torch.cat([composite_color, composite_alpha], dim=1)
    return composite_rgba

def sample_binned_uniform_distances(t_min, t_max, num_rays, num_samples_per_ray):
    """
    Sample distances along rays in a uniform grid
    """
    dt = t_max - t_min
    bin_dt = dt / num_samples_per_ray
    u = torch.rand((num_rays, num_samples_per_ray)) * bin_dt
    t = torch.linspace(t_min, t_max - bin_dt, num_samples_per_ray) + u
    return t

def get_cam_ray_dirs(width, height,
                     vertical_fov_radians):
    """
    Get the intrinsic camera ray directions
    """
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
    return cam_ray_dir

def get_ray_dirs(cam_ray_dir,
                 camera_local_to_world):
    """
    Transform camera intrinsic ray directions
    to world space directions
    """
    height, width, _ = cam_ray_dir.shape
    camera_local_to_world_rot = camera_local_to_world[:3, :3]
    ray_dir = torch.matmul(
        camera_local_to_world_rot,
        cam_ray_dir.view(height * width, 3, 1))
    ray_dir = ray_dir.view(height, width, 3)
    return ray_dir

def sphere_radiance(position, view_direction):
    """
    position is (N, 3)
    view_direction is (N, 3)
    returns (N, 1) density and (N, 3) color
    """
    center = torch.tensor([0.0, 0.0, -5.0])
    radius = 1.0
    distance_to_surface = torch.norm(position - center.view(1, 3), dim=1, keepdim=True) - radius
    density = torch.clamp(torch.relu(-distance_to_surface), 0.0, 0.1) * 10
    color = torch.clamp(position - center.view(1, 3), 0, 1)
    return (density, color)

def show_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # remove the axis
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.show()

if __name__ == "__main__":
    # for angle in [0]:
    for angle in [-30, 0, 30]:
        y_rot_rads = math.radians(angle)
        camera_local_to_world=torch.eye(4)
        camera_local_to_world[0, 0] = math.cos(y_rot_rads)
        camera_local_to_world[0, 2] = math.sin(y_rot_rads)
        camera_local_to_world[2, 0] = -math.sin(y_rot_rads)
        camera_local_to_world[2, 2] = math.cos(y_rot_rads)

        show_image(
            render(
                sphere_radiance,
                360, 240,
                math.radians(60.0),
                z_near=3.0, z_far=7.0,
                num_samples_per_ray=11,
                camera_local_to_world=camera_local_to_world))

