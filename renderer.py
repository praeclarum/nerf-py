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

def render(radiance, width, height,
           vertical_fov, z_near, z_far, camera_pose):
    # camera_pose is homogenous 4x4 therefore the
    # camera position is the last column
    camera_position = camera_pose[:3, 3]
    print(camera_position.shape)

    # compute the camera view direction
    # this is the negative z-axis of the camera
    # in world coordinates
    camera_view_direction = -camera_pose[:3, 2]

    # compute the camera up direction
    # this is the y-axis of the camera
    # in world coordinates
    camera_up_direction = camera_pose[:3, 1]

    # compute the camera right direction
    # this is the x-axis of the camera
    # in world coordinates
    camera_right_direction = camera_pose[:3, 0]

    # compute the camera vertical field of view
    # in radians
    vertical_fov_radians = vertical_fov

    # compute the camera horizontal field of view
    # in radians
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
    print(cam_ray_dir.shape, cam_ray_dir)

    colors = torch.zeros((height, width, 3), dtype=torch.float32)
    colors[:, :, 0] = 1.0
    return colors

def show_image(image):
    plt.imshow(image)
    plt.show()

show_image(
    render(
        None,
        7, 5,
        math.radians(60.0),
        z_near=0.1, z_far=100.0,
        camera_pose=torch.eye(4)))    

