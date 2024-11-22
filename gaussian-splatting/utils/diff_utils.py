import torch


# 1. Define the function to convert Cartesian to spherical coordinates
def transform_spherical(cartesian_coords):
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z / r)
    phi = torch.arctan2(y, x)
    res = torch.stack([r, theta, phi], dim=1)

    return res

# 2. Manually compute the Jacobian for the transformation from Cartesian to spherical coordinates
def jacobian_spherical(cartesian_coords):
    device = cartesian_coords.device
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    num_samples = cartesian_coords.shape[0]
    res = torch.empty((num_samples, 3, 3)).to(device)

    r2 = x**2 + y**2 + z**2
    r = torch.sqrt(r2)

    res[:, 0, 0] = x / r
    res[:, 0, 1] = y / r
    res[:, 0, 2] = z / r
    res[:, 1, 0] = x * z/(torch.sqrt(-z**2/r2 + 1)*r2**(3/2))
    res[:, 1, 1] = y*z/(torch.sqrt(-z**2/r2 + 1)*r2**(3/2))
    res[:, 1, 2] = -(-z**2/r2**(3/2) + 1/r)/torch.sqrt(-z**2/r2 + 1)
    res[:, 2, 0] = -y/(x**2 + y**2)
    res[:, 2, 1] = x/(x**2 + y**2)
    res[:, 2, 2] = 0

    return res

def jacobian_all(cartesian_coords, polyvals):
    spherical_coords = transform_spherical(cartesian_coords)
    polar_coords = transform_polar(spherical_coords, polyvals)

    J_spherical = jacobian_spherical(cartesian_coords)
    J_polar = jacobian_polar(spherical_coords, polyvals)
    J_cartesian = jacobian_cartesian(polar_coords)

    J_all = J_cartesian @ J_polar @ J_spherical

    # Replace central points with identity matrix
    magnitude_ratio = torch.sqrt(torch.sum(cartesian_coords[:, :2]**2, dim=1)) / torch.sqrt(cartesian_coords[:, 2]**2) 
    rounding_mask = magnitude_ratio < 1e-3

    # Replace all degenerate jac matrices with identity matrix
    is_degen = torch.linalg.det(J_all) < 1e-3
    is_degen[torch.isnan(is_degen)] = True

    # Replace nans and infs with identity matrix
    nan_mask = torch.any(torch.isnan(J_all), dim=(1,2))
    inf_mask = torch.any(torch.isinf(J_all), dim=(1,2))
    mask = nan_mask | inf_mask | rounding_mask | is_degen
    J_all[mask] = torch.eye(3, device=J_all.device)

    return J_all

def poly(phi, polyvals):
    return sum(c * phi**i for i, c in enumerate(polyvals))
    #return polyvals[0] + polyvals[1] * phi + polyvals[2] * phi**2

def poly_prime(phi, polyvals):
    return sum(i * polyvals[i] * phi**(i-1) for i in range(1, len(polyvals)))
    # return polyvals[1] + 2 * polyvals[2] * phi

def transform_polar(spherical_coords, polyvals):
    r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
    theta = poly(theta, polyvals)

    res = torch.stack([r, theta, phi], dim=1)

    return res

def jacobian_polar(spherical_coords, polyvals):
    device = spherical_coords.device
    theta = spherical_coords[:, 1]

    res = torch.zeros((spherical_coords.shape[0], 3, 3)).to(device)
    res[:, 0, 0] = 1
    res[:, 1, 1] = poly_prime(theta, polyvals)
    res[:, 2, 2] = 1

    return res

def transform_cartesian(spherical_coords):
    r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    res = torch.stack([x, y, z], dim=1)

    return res

def jacobian_cartesian(spherical_coords):
    device = spherical_coords.device
    r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
    
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)
    sinphi = torch.sin(phi)
    cosphi = torch.cos(phi)
    
    res = torch.empty((spherical_coords.shape[0], 3, 3)).to(device)
    res[:, 0, 0] = sintheta * cosphi
    res[:, 0, 1] = r * cosphi * costheta
    res[:, 0, 2] = -r * sinphi * sintheta
    res[:, 1, 0] = sinphi * sintheta
    res[:, 1, 1] = r * sinphi * costheta
    res[:, 1, 2] = r * sintheta * cosphi
    res[:, 2, 0] = costheta
    res[:, 2, 1] = -r * sintheta
    res[:, 2, 2] = 0

    return res