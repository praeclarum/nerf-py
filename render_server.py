from flask import Flask, request, send_file
from PIL import Image
import io
import os
import sys
import datetime
import shutil
import glob
import torch
import numpy as np
import math
import data
import model
import renderer
from PIL import Image


def load_checkpoint(path):
    checkpoint = torch.load(path)
    num_trained_steps = checkpoint["num_trained_steps"]
    nerf_model = model.MildenhallNeRF(
        include_view_direction=include_view_direction, device=device
    ).to(device)
    nerf_model.load_state_dict(checkpoint["model_state_dict"])
    run_id = os.path.basename(os.path.dirname(path))
    return nerf_model, run_id, num_trained_steps


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

include_view_direction = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
print(f'Using device "{device}": {device_name}')


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>NeRF Render Server</title>
</head>
<body>
    <h1>NeRF Render Server</h1>
    <p>Send a POST request to /api/generate with a JSON body containing a 4x4 matrix.</p>
    <p>The last column should be <code>[Tx, Ty, Tz, 1]</code>.</p>
    <p>Example:</p>
    <code><pre>
{
    "matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
    </pre></code>
    <p>Response:</p>
    <img src="/api/generate" />
</body>
</html>"""


@app.route("/api/generate", methods=["POST", "GET"])
def generate_image():
    # data = request.json
    # matrix = data["matrix"]

    image = images[0]

    # Convert to tensor and make sure it's 4x4
    # cam_transform = torch.tensor(matrix, dtype=torch.float32).view(4, 4)
    cam_transform = image.extrinsics
    cam_ray_dirs = image.cam_ray_dirs

    # Generate the image using your model
    generated_image = render_image(cam_ray_dirs=cam_ray_dirs, cam_transform=cam_transform)

    # Convert the output tensor to a numpy array, and then to PIL image
    generated_image = np.clip(
        generated_image.detach().cpu().numpy() * 255, 0, 255
    ).astype(np.uint8)
    image = Image.fromarray(generated_image)

    # Write to a BytesIO stream
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    stream.seek(0)

    return send_file(stream, mimetype="image/jpeg")


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    images_dir = f"/Volumes/home/Data/datasets/nerf/{dataset_name}"
    # image = data.ImageInfo(images_dir, "Frame0", 128, device)
    # print(f"IMAGE WIDTH {image.width}, HEIGHT {image.height}")
    # train_image = image.image_tensor.to(device)
    images = data.load_images(images_dir, 128, device)

    nerf_model, run_id, num_train_steps = load_checkpoint(
        "/Volumes/nn/Data/generated/nerf/piano3/2023-08-02_16-11-02/checkpoint_35716.pth"
    )

    app.run(host="0.0.0.0", debug=True)
