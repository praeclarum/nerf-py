from flask import Flask, request, send_file
from PIL import Image
import io
import os
import sys
import torch
import numpy as np
import math
import data
import model
import renderer
import gc


def load_checkpoint(path):
    global num_trained_steps, run_id, z_near, z_far, bb_min, bb_max, include_view_direction, cam_transforms, train_num_samples_per_ray
    checkpoint = torch.load(path)
    num_trained_steps = checkpoint["num_trained_steps"]
    include_view_direction = checkpoint["include_view_direction"]
    cam_transforms = [
        torch.tensor(x, device=device) for x in checkpoint["cam_transforms"]
    ]
    train_num_samples_per_ray = checkpoint["train_num_samples_per_ray"]
    bb_min = torch.tensor(checkpoint["bb_min"], device=device)
    bb_max = torch.tensor(checkpoint["bb_max"], device=device)
    z_near = checkpoint["z_near"]
    z_far = checkpoint["z_far"]
    print(
        f"bb_min: {bb_min.tolist()}, bb_max: {bb_max.tolist()}, z_near: {z_near}, z_far: {z_far}, include_view_direction: {include_view_direction}"
    )
    print(
        f"z_near: {z_near}, z_far: {z_far}, include_view_direction: {include_view_direction}"
    )
    print(f"include_view_direction: {include_view_direction}")
    m = model.MildenhallNeRF(
        include_view_direction=include_view_direction,
        bb_min=bb_min,
        bb_max=bb_max,
        device=device,
    ).to(device)
    m.load_state_dict(checkpoint["model_state_dict"])
    run_id = os.path.basename(os.path.dirname(path))
    return m


def render_image(cam_ray_dirs, cam_transform):
    with torch.no_grad():
        m = get_model()
        return renderer.render_image(
            m,
            cam_ray_dirs,
            z_near=z_near,
            z_far=z_far,
            num_samples_per_ray=train_num_samples_per_ray,
            camera_local_to_world=cam_transform,
            include_view_direction=include_view_direction,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
print(f'Using device "{device}": {device_name}')

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    m = get_model()
    initial_matrix = (
        cam_transforms[torch.randint(len(cam_transforms), (1,))]
        if len(cam_transforms) > 0
        else torch.eye(4, device=device)
    )
    initial_matrix = initial_matrix.tolist()
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{dataset_name} - {run_id}</title>
</head>
<body>
    <h1>{dataset_name} - {run_id}</h1>
    <img id="rendererOutput" src="/api/render" width="384" />
    <script src="/renderer.js"></script>
    <script>
        const initialMatrix = {initial_matrix};
        startRenderer(document.getElementById("rendererOutput"), initialMatrix);
    </script>
</body>
</html>"""


@app.route("/renderer.js", methods=["GET"])
def renderer_js():
    return send_file("renderer.js")


@app.route("/api/render", methods=["POST", "GET"])
def generate_image():
    data = request.json

    # Convert to tensor and make sure it's 4x4
    cam_transform = torch.tensor(
        data["cam_transform"], dtype=torch.float32, device=device
    ).view(4, 4)

    # Generate the image using your model
    height = 256
    width = 256  # int(height * aspect)
    horizontal_fov_degrees = 62.3
    cam_ray_dirs = renderer.get_fov_cam_ray_dirs(
        width,
        height,
        horizontal_fov_radians=math.radians(horizontal_fov_degrees),
        device=device,
    )

    rendered_image = render_image(
        cam_ray_dirs=cam_ray_dirs, cam_transform=cam_transform
    )

    # Convert the output tensor to a numpy array, and then to PIL image
    image_ar = np.clip(rendered_image.detach().cpu().numpy() * 255, 0, 255).astype(
        np.uint8
    )
    image = Image.fromarray(image_ar)
    image = image.rotate(270, expand=True)

    # Write to a BytesIO stream
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    stream.seek(0)

    del rendered_image
    del cam_ray_dirs

    return send_file(stream, mimetype="image/jpeg")


def get_images():
    global images
    if images is None:
        images = data.load_images(images_dir, 16, device)
    return images


def get_model():
    global nerf_model
    if nerf_model is None:
        nerf_model = load_checkpoint(checkpoint_path)
        nerf_model.eval()
    return nerf_model


images = None
nerf_model = None
train_num_samples_per_ray = 16
z_near = 0.01
z_far = 4.0
bb_min = None
bb_max = None
include_view_direction = True
run_id = None
cam_transforms = []

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    run_id = os.path.basename(os.path.dirname(checkpoint_path))
    images_dir = f"/Volumes/home/Data/datasets/nerf/{dataset_name}"
    app.run(host="0.0.0.0", debug=True)
