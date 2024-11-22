from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
from utils.general_utils import build_convariance_matrix, build_rotation, strip_symmetric
import scipy.spatial.transform as st

def upload_image(image, topic):
    image = Image.fromarray(image)
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    url = "http://127.0.0.1:5000/post_image"
    files = {'file': ('image.jpeg', image_io, 'image/jpeg')}
    data = {'topic': topic}
    try:
        response = requests.post(url, files=files, data=data)

        if response.status_code != 200:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send image: {e}")


if __name__ == "__main__":
    fx = 256
    fy = 256
    cx = 256
    cy = 256
    width = 512
    height = 512
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
    fovx = 2 * np.arctan(cx / (2 * fx))
    fovy = 2 * np.arctan(cy / (2 * fy))

    tanfovx = np.tan(fovx / 2)
    tanfovy = np.tan(fovy / 2)

    bg_color = torch.tensor([0.0, 0.0, 0.0]).float().cuda()

    scaling_modifier = 1.0

    projectionMatrix = getProjectionMatrix(
        znear=float(0.1), 
        zfar=float(10.0), 
        fovX=fovx, fovY=fovy).transpose(0,1).cuda()

    raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=torch.eye(4).cuda(),
            projmatrix=projectionMatrix,
            sh_degree=2,
            campos=torch.zeros(3).cuda(),
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = torch.tensor([
        [0.0, -1.0, 5.0],
        [0.0, 1.0, 5.0]
    ]).float().cuda()
    means2D = torch.zeros(2, 2).float().cuda()
    shs = torch.zeros(2, 16, 3).float().cuda()
    shs[0, 0, 0] = 1.0
    shs[1, 0, 1] = 1.0
    scales = torch.tensor([
        [0.2, 1.0, 0.1],
        [0.2, 1.0, 0.1]
    ]).float().cuda()
    rotations = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ]).float().cuda()

    # Build covariance matrix
    S = torch.zeros((2, 3, 3)).float().cuda()
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    euls = np.array([-12.0, 5.0, 72.0])
    R_np = st.Rotation.from_euler('xyz', euls, degrees=True).as_matrix()
    R_np = np.stack((R_np, ) * 2, axis=0)
    R = torch.tensor(R_np).float().cuda()
    cov3D = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
    cov3D = strip_symmetric(cov3D)

    opacity = torch.tensor([[1.0], [1.0]]).float().cuda()
    colors_precomp = None

    kwargs = {
            "means3D": means3D,
            "means2D": means2D,
            "shs": shs,
            "colors_precomp": colors_precomp,
            "opacities": opacity,
            "scales": None, #scales,
            "rotations": None, #rotations,
            "cov3D_precomp": cov3D,
        }

    for key, value in kwargs.items():
        if value is not None:
            print(f"{key}: {value.dtype}")


    rendered_image, radii, depth = rasterizer(**kwargs)

    image = rendered_image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)

    upload_image(image, "cov_precomp")


    # Modify scale and rotation for ordered scales
    scales_idx = torch.argsort(scales).flip(-1)
    scales_sorted = torch.gather(scales, dim=-1, index=scales_idx)
    S_sorted = torch.zeros_like(S)
    S_sorted[:, 0, 0] = scales_sorted[:, 0]
    S_sorted[:, 1, 1] = scales_sorted[:, 1]
    S_sorted[:, 2, 2] = scales_sorted[:, 2]

    R_idx = scales_idx[:, None, :].expand(-1, 3, -1)
    R_sorted = torch.gather(R, dim=2, index=R_idx)
    dets = torch.det(R_sorted)
    R_sorted *= torch.sign(dets)[:, None, None]

    # Build covariance matrix from modified scale and rotation
    cov3D_sorted = R_sorted @ S_sorted @ S_sorted.transpose(1, 2) @ R_sorted.transpose(1, 2)
    cov3D_sorted = strip_symmetric(cov3D_sorted)

    cov3D_error = torch.linalg.norm(cov3D - cov3D_sorted)
    print(f"Error: {cov3D_error:.1e}")