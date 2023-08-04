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
    checkpoint = torch.load(path)
    num_trained_steps = checkpoint["num_trained_steps"]
    images = get_images()
    bb_min, bb_max = data.get_images_bounding_box(images)
    m = model.MildenhallNeRF(
        include_view_direction=include_view_direction,
        bb_min=bb_min,
        bb_max=bb_max,
        device=device,
    ).to(device)
    m.load_state_dict(checkpoint["model_state_dict"])
    run_id = os.path.basename(os.path.dirname(path))
    return m, run_id, num_trained_steps


def render_image(cam_ray_dirs, cam_transform, num_samples_per_ray=32):
    with torch.no_grad():
        return renderer.render_image(
            get_model(),
            cam_ray_dirs,
            z_near=0.01,
            z_far=3.0,
            num_samples_per_ray=num_samples_per_ray,
            camera_local_to_world=cam_transform,
            include_view_direction=include_view_direction,
        )


include_view_direction = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
print(f'Using device "{device}": {device_name}')


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    images = get_images()
    initial_image = images[torch.randint(len(images), (1,))]
    initial_matrix = initial_image.extrinsics.cpu().numpy().tolist()
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
        nerf_model, run_id, num_train_steps = load_checkpoint(checkpoint_path)
        nerf_model.eval()
    return nerf_model


if __name__ == "__main__":
    checkpoint_path = sys.argv[1]

    # "/Volumes/nn/Data/generated/nerf/piano3/2023-08-03_12-07-32/checkpoint_4034.pth"

    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    run_id = os.path.basename(os.path.dirname(checkpoint_path))
    images_dir = f"/Volumes/home/Data/datasets/nerf/{dataset_name}"
    images = None
    nerf_model = None

    app.run(host="0.0.0.0", debug=True)
