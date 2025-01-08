import numpy as np
import os


def quat_to_rotm(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    N = len(w)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return R


def make_K(intrinsics):
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = intrinsics["fx"]
    K[1, 1] = intrinsics["fy"]
    K[0, 2] = intrinsics["cx"]
    K[1, 2] = intrinsics["cy"]
    K[2, 2] = 1
    return K

def spatial_distortion(points, intrinsics):
    # Perform fisheye distortion
    x, y, z = points
    dist_to_origin = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)

    # for i in [3]:
    #     intrinsics["distortion_params"][i] = 0

    theta_mod = theta * (1 + intrinsics["distortion_params"][0]*theta**2 + intrinsics["distortion_params"][1]*theta**4 + intrinsics["distortion_params"][2]*theta**6 + intrinsics["distortion_params"][3]*theta**8)

    x_post = theta_mod * np.sin(np.atan2(x, y)) # (x / np.sqrt(x**2 + y**2))
    y_post = theta_mod * np.cos(np.atan2(x, y)) # (y / np.sqrt(x**2 + y**2))
    z_post = np.ones_like(x_post)
    points_post = np.vstack((x_post, y_post, z_post))
    points_post *= (dist_to_origin / np.sqrt(x_post**2 + y_post**2 + z_post**2))

    return points_post


def to_spherical(points3D):
    x, y, z = points3D
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.vstack((r, theta, phi))

def to_cartesian(spherical):
    r, theta, phi = spherical
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack((x, y, z))

def approx_distortion_poly(cameras_dict, fisheye_poly_degree=8):
    cameras_dict_out = {}

    for camera_id, intrinsics in cameras_dict.items():
        # Generate points on a half-circle within the yz-plane
        N = 100
        alpha = np.linspace(0, np.pi, N)
        x_pre = np.zeros(N)
        y_pre = np.sin(alpha)
        z_pre = np.cos(alpha)
        points_pre = np.vstack((x_pre, y_pre, z_pre))

        # Distort points
        spherical_pre = to_spherical(points_pre)
        points_post = spatial_distortion(points_pre, intrinsics)
        spherical_post = to_spherical(points_post)

        # Check for violation of monotonicity
        theta_pre = spherical_pre[1]
        theta_post = spherical_post[1]
        indices_violoation = np.where(np.diff(theta_post) < 0)[0]
        filter_index = len(theta_post)
        max_theta_pre = np.pi
        if len(indices_violoation) > 0:
            filter_index = indices_violoation[0]
            max_theta_pre = theta_pre[filter_index]
        theta_post_filtered = theta_post[:filter_index]
        theta_pre_filtered = theta_pre[:filter_index]

        # Remove nans
        isnan = np.isnan(theta_post_filtered) | np.isnan(theta_pre_filtered)
        theta_post_filtered = theta_post_filtered[~isnan]
        theta_pre_filtered = theta_pre_filtered[~isnan]

        # Fit polynomial
        coeffs = np.polyfit(theta_pre_filtered, theta_post_filtered, fisheye_poly_degree)[::-1]
        theta_post_fit = sum(ci * theta_pre_filtered**i for i, ci in enumerate(coeffs))
        mse = np.mean((theta_post_filtered - theta_post_fit)**2)

        # Save results
        cameras_dict_out[camera_id] = intrinsics.copy()
        cameras_dict_out[camera_id]["distortion_params"] = coeffs
        cameras_dict_out[camera_id]["mse"] = mse
        cameras_dict_out[camera_id]["max_fov_monotonic"] = max_theta_pre * 2
        cameras_dict_out[camera_id]["max_fov_monotonic_deg"] = np.rad2deg(max_theta_pre) * 2

    return cameras_dict_out