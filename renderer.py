import math
import torch
from torch import nn
import matplotlib.pyplot as plt


def render_image(
    radiance,
    cam_ray_dirs,
    z_near,
    z_far,
    num_samples_per_ray,
    camera_local_to_world,
    include_view_direction,
):
    """
    Render a `radiance` function

    `radiance(position_and_view_direction) -> density_and_color`

    where `position_and_view_direction` are shaped (N, 6),
    `density_and_color` is shaped (N, 4), and N is the radiance batch size.

    `z_near` and `z_far` are the near and far clipping planes.

    `cam_ray_dirs` is shaped (height, width, 3) and are the intrinsic camera ray directions.

    `num_samples_per_ray` is the number of random points to sample when
    calculating the output color.

    Position and orientation of the camera represented as a 4x4 matrices shaped (B, 4, 4)
    where B is the batch size.
    """
    height, width, _ = cam_ray_dirs.shape
    num_rays = width * height
    ray_origins = camera_local_to_world[:3, 3].view(1, 3)
    ray_dirs = get_ray_dirs(cam_ray_dirs.reshape(num_rays, 3), camera_local_to_world)
    ray_colors = render_rays(
        radiance,
        ray_origs=ray_origins,
        ray_dirs=ray_dirs,
        z_near=z_near,
        z_far=z_far,
        num_samples_per_ray=num_samples_per_ray,
        include_view_direction=include_view_direction,
    )
    colors = ray_colors.view(height, width, 3)
    return colors


def render_rays(
    radiance,
    ray_origs,
    ray_dirs,
    z_near,
    z_far,
    num_samples_per_ray,
    include_view_direction,
) -> torch.Tensor:
    """
    Render a `radiance` function

    `radiance(position_and_view_direction) -> density_and_color`

    where `position_and_view_direction` are shaped (N, 6),
    `density_and_color` is shaped (N, 4), and N is the radiance batch size.

    * `ray_origins` is (num_rays or 1, 3) points
    * `ray_dirs` is (num_rays, 3) normalized world vectors
    * output is (num_rays, 3) colors
    * `z_near` and `z_far` are the near and far clipping planes.
    """
    num_rays = ray_dirs.shape[0]
    # Get the sample points along each ray
    sample_distances = sample_log_binned_uniform_distances(
        z_near, z_far, num_rays, num_samples_per_ray, device=ray_origs.device
    )
    sample_distances = sample_distances.view(num_rays, num_samples_per_ray, 1)
    ray_origs = ray_origs.unsqueeze(1)
    ray_dirs = ray_dirs.unsqueeze(1)
    sample_positions = ray_dirs * sample_distances + ray_origs
    sample_dirs = ray_dirs.repeat(1, num_samples_per_ray, 1)
    sample_positions_and_dirs = sample_positions.view(num_rays * num_samples_per_ray, 3)
    if include_view_direction:
        sample_positions_and_dirs = torch.cat(
            [
                sample_positions.view(num_rays * num_samples_per_ray, 3),
                sample_dirs.view(num_rays * num_samples_per_ray, 3),
            ],
            dim=1,
        )

    # Get the density and color at each sample point
    sample_densities_and_colors = radiance(sample_positions_and_dirs)
    sample_densities = sample_densities_and_colors[:, 0].view(
        num_rays, num_samples_per_ray
    )
    sample_colors = sample_densities_and_colors[:, 1:].view(
        num_rays, num_samples_per_ray, 3
    )

    # Integrate the density and color along the ray
    mean_sample_separation = (z_far - z_near) / num_samples_per_ray
    colors = alpha_composite_rgb(
        sample_densities,
        sample_colors,
        mean_sample_separation,
        sample_distances.view(num_rays, num_samples_per_ray),
    )
    return colors


def alpha_composite_rgb(
    sample_densities, sample_colors, mean_sample_separation, sample_distances
):
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
    composite_alpha = 1.0 - torch.exp(
        -sample_densities[:, -1:] * mean_sample_separation
    )
    composite_color = composite_alpha * sample_colors[:, -1]
    for i in range(num_samples_per_ray - 2, -1, -1):
        alpha_i = torch.unsqueeze(alphas[:, i], 1)
        one_minus_alpha_i = 1.0 - alpha_i
        composite_color = (
            alpha_i * sample_colors[:, i, :] + one_minus_alpha_i * composite_color
        )
        # composite_alpha = alpha_i + one_minus_alpha_i * composite_alpha
    # composite_rgba = torch.cat([composite_color, composite_alpha], dim=1)
    # return composite_rgba
    return composite_color


def alpha_composite_rgba(
    sample_densities, sample_colors, mean_sample_separation, sample_distances
):
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
    composite_alpha = 1.0 - torch.exp(
        -sample_densities[:, -1:] * mean_sample_separation
    )
    composite_color = sample_colors[:, -1]
    for i in range(num_samples_per_ray - 2, -1, -1):
        alpha_i = torch.unsqueeze(alphas[:, i], 1)
        one_minus_alpha_i = 1.0 - alpha_i
        composite_color = (
            alpha_i * sample_colors[:, i, :] + one_minus_alpha_i * composite_color
        )
        composite_alpha = alpha_i + one_minus_alpha_i * composite_alpha
    composite_rgba = torch.cat([composite_color, composite_alpha], dim=1)
    return composite_rgba


def sample_linear_binned_uniform_distances(
    t_min, t_max, num_rays, num_samples_per_ray, device
):
    """
    Sample distances along rays in a uniform grid
    """
    dt = t_max - t_min
    bin_dt = dt / num_samples_per_ray
    u = torch.rand((num_rays, num_samples_per_ray), device=device) * bin_dt
    t = torch.linspace(t_min, t_max - bin_dt, num_samples_per_ray, device=device) + u
    return t


def sample_log_binned_uniform_distances(
    t_min, t_max, num_rays, num_samples_per_ray, device
):
    """
    Sample distances along rays in a uniform grid
    """
    log_t_max = math.log(t_max)
    log_t_min = math.log(t_min)
    log_dt = log_t_max - log_t_min
    log_bin_dt = log_dt / num_samples_per_ray
    log_u = torch.rand((num_rays, num_samples_per_ray), device=device) * log_bin_dt
    log_t = (
        torch.linspace(
            log_t_min, log_t_max - log_bin_dt, num_samples_per_ray, device=device
        )
        + log_u
    )
    t = torch.exp(log_t)
    return t


def get_fov_cam_ray_dirs(width, height, horizontal_fov_radians, device):
    """
    Get the intrinsic camera ray directions
    """
    # compute ray directions for every pixel in camera space
    rot_y_rads = torch.linspace(
        horizontal_fov_radians / 2.0,
        -horizontal_fov_radians / 2.0,
        width,
        device=device,
    )
    vertical_fov_radians = horizontal_fov_radians * height / width
    rot_x_rads = torch.linspace(
        vertical_fov_radians / 2.0, -vertical_fov_radians / 2.0, height, device=device
    )
    rot_y_rads = rot_y_rads.view(1, width).repeat(height, 1)
    rot_x_rads = rot_x_rads.view(height, 1).repeat(1, width)
    cam_ray_x_dir = -torch.sin(rot_y_rads) * torch.cos(rot_x_rads)
    cam_ray_y_dir = torch.sin(rot_x_rads)
    cam_ray_z_dir = -torch.cos(rot_y_rads) * torch.cos(rot_x_rads)
    # cam_ray_z_dir = -torch.sqrt((1.0 - torch.square(cam_ray_x_dir) - torch.square(cam_ray_y_dir)))
    cam_ray_dir = torch.stack([cam_ray_x_dir, cam_ray_y_dir, cam_ray_z_dir], dim=2)
    return cam_ray_dir


def get_intrinsic_cam_ray_dirs(width, height, intrinsics, offset, device):
    image_xis = (
        torch.linspace(0, width - 1, width, device=device)
        .reshape(1, width)
        .repeat(height, 1)
    )
    image_yis = (
        torch.linspace(0, height - 1, height, device=device)
        .reshape(height, 1)
        .repeat(1, width)
    )
    # print("IMAGE XYIS", image_xyis.shape, image_xyis)
    dir_xs = (
        (image_xis[:, :] + offset - intrinsics[0, 2]) / intrinsics[0, 0]
    ).unsqueeze(-1)
    dir_ys = (
        -(image_yis[:, :] + offset - intrinsics[1, 2]) / intrinsics[1, 1]
    ).unsqueeze(-1)
    dir_zs = -torch.ones_like(dir_xs)
    # print("IMAGE XS", dir_xs.shape)
    # print("IMAGE YS", dir_ys.shape)
    # print("IMAGE ZS", dir_zs.shape)
    dir_xyzs = torch.cat([dir_xs, dir_ys, dir_zs], dim=-1)
    # print("IMAGE XYZS", dir_xyzs.shape)
    # Normalize
    intrinsic_dirs = dir_xyzs / torch.norm(dir_xyzs, dim=-1, keepdim=True)
    # print("INTRINSIC DIRS", intrinsic_dirs.shape)
    # print("INTR0", intrinsic_dirs[0, 0, :])
    return intrinsic_dirs.contiguous()


def get_ray_dirs(cam_ray_dir: torch.Tensor, camera_local_to_world):
    """
    Transform camera intrinsic ray directions
    to world space directions

    * `cam_ray_dir` is (N, 3)
    * `camera_local_to_world` is (4, 4)
    * output is (N, 3)
    """
    # print("GET CAM RAY DIR", cam_ray_dir.shape)
    cam_ray_dir = cam_ray_dir.unsqueeze(-1)
    # print("GET CAM RAY DIR V", cam_ray_dir.shape)
    camera_local_to_world_rot = camera_local_to_world[:3, :3]
    ray_dir = torch.matmul(camera_local_to_world_rot, cam_ray_dir)
    ray_dir = ray_dir.squeeze(-1)
    return ray_dir


def show_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image.detach().numpy())
    ax.axis("off")  # remove the axis
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.show()
    plt.close()


if __name__ == "__main__":

    def sphere_radiance(position_and_view_direction):
        """
        `position` is (N, 3),
        `view_direction` is (N, 3).

        Returns (N, 4) density and color
        """
        center = torch.tensor([0.0, 0.0, -5.0])
        radius = 1.0
        position = position_and_view_direction[:, :3]
        distance_to_surface = (
            torch.norm(position - center.view(1, 3), dim=1, keepdim=True) - radius
        )
        density = torch.clamp(torch.relu(-distance_to_surface), 0.0, 0.1) * 10
        color = torch.clamp(position - center.view(1, 3), 0, 1)
        return torch.cat([density, color], dim=1)

    # sphere_radiance(torch.tensor([[0.1, 0.0, -4.5, 0.0, 0.0, 1.0]]))

    cam_ray_dirs = get_fov_cam_ray_dirs(700, 500, math.radians(60.0), "cpu")

    def renderer(cam):
        return render_image(
            sphere_radiance,
            cam_ray_dirs,
            z_near=3.0,
            z_far=7.0,
            num_samples_per_ray=11,
            camera_local_to_world=cam,
            include_view_direction=True,
        )

    for angle in [-20, 0, 20]:
        y_rot_rads = math.radians(angle)
        camera_local_to_world = torch.eye(4)
        camera_local_to_world[0, 0] = math.cos(y_rot_rads)
        camera_local_to_world[0, 2] = math.sin(y_rot_rads)
        camera_local_to_world[2, 0] = -math.sin(y_rot_rads)
        camera_local_to_world[2, 2] = math.cos(y_rot_rads)
        show_image(
            renderer(camera_local_to_world),
        )
