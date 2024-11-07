import torch
from utils.diff_utils import jacobian_all
import configparser
import numpy as np
import matplotlib.pyplot as plt
import tqdm

dump_file = "jac_degen.dump"
config = configparser.ConfigParser()
config.read("control.ini")
poly_coeffs = [float(config["mods"][f"theta_mod_{i}"]) for i in range(5)]
means3D = torch.load(dump_file, weights_only=True)

J = jacobian_all(means3D[None, :], poly_coeffs)
