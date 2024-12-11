import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

formats = ["svg", "png", "pdf"]

plt.rcParams["font.family"] = "Times New Roman"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    # Remove trailing slash
    args.results_dir = args.results_dir.rstrip("/")

    # Load the polydegree results
    df_polydegree = pd.read_csv(os.path.join(args.results_dir, "polydegree.csv"))

    # Plot the results using twinx (shared x-axis for all, left y axis (psnr), right y axis (ssim, lpips))
    fig, ax1 = plt.subplots()

    # Set the size to 6x4
    fig.set_size_inches(5, 3)

    # Plot the polydegree results
    ax1.plot(df_polydegree["polydegree"], df_polydegree["psnr"], label="PSNR", color="tab:blue", alpha=0.3)
    ax1.plot(df_polydegree["polydegree"], df_polydegree["psnr"], label="PSNR", color="tab:blue", marker="o", linestyle="")
    ax1.set_xlabel("Polynomial Degree")
    ax1.set_ylabel("PSNR", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_ylim([27, 29.5])

    ax2 = ax1.twinx()
    ax2.plot(df_polydegree["polydegree"], df_polydegree["ssim"], label="SSIM", color="tab:orange", alpha=0.3)
    ax2.plot(df_polydegree["polydegree"], df_polydegree["ssim"], label="SSIM", color="tab:orange", marker="o", linestyle="")
    ax2.set_ylabel("SSIM", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    ax2.set_ylim([0.9, 0.93])

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(df_polydegree["polydegree"], df_polydegree["lpips"], label="LPIPS", color="tab:green", alpha=0.3)
    ax3.plot(df_polydegree["polydegree"], df_polydegree["lpips"], label="LPIPS", color="tab:green", marker="o", linestyle="")
    ax3.set_ylabel("LPIPS", color="tab:green")
    ax3.tick_params(axis='y', labelcolor="tab:green")
    ax3.set_ylim([0.15, 0.18])
    
    # Save the plot
    output_dir = os.path.join("figures", os.path.basename(args.results_dir))
    for fmt in formats:
        plt.savefig(os.path.join(output_dir, f"polydegree.{fmt}"), format=fmt, bbox_inches="tight")
    
    # Compute sensitivity for polydegree=8
    sen_degree = 8
    iloc = df_polydegree["polydegree"].tolist().index(sen_degree)
    psnr_self = df_polydegree["psnr"].iloc[iloc]
    psnr_high = df_polydegree["psnr"].iloc[iloc+1]
    psnr_low = df_polydegree["psnr"].iloc[iloc-1]
    ssim_self = df_polydegree["ssim"].iloc[iloc]
    ssim_high = df_polydegree["ssim"].iloc[iloc+1]
    ssim_low = df_polydegree["ssim"].iloc[iloc-1]
    lpips_self = df_polydegree["lpips"].iloc[iloc]
    lpips_high = df_polydegree["lpips"].iloc[iloc+1]
    lpips_low = df_polydegree["lpips"].iloc[iloc-1]

    sen_psnr = 0.5 * (np.abs(psnr_high - psnr_self) + np.abs(psnr_self - psnr_low))
    sen_ssim = 0.5 * (np.abs(ssim_high - ssim_self) + np.abs(ssim_self - ssim_low))
    sen_lpips = 0.5 * (np.abs(lpips_high - lpips_self) + np.abs(lpips_self - lpips_low))

    print(f"Sensitivity for polydegree={sen_degree}: PSNR={sen_psnr:.2e}, SSIM={sen_ssim:.2e}, LPIPS={sen_lpips:.2e}")
