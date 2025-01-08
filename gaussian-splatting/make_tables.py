import pandas as pd
import argparse
import os
import numpy as np

decimals_psnr = 2
decimals_ssim = 3
decimals_lpips = 3

pretty_names = {
    "archiviz": "Archiviz",
    "barbershop": "Barbershop",
    "classroom": "Classroom",
    "monk": "Monk",
    "pabellon": "Pabellon",
    "sky": "Sky",
    "bedroom": "Bedroom",
    "kitchen": "Kitchen",
    "office_day": "Office Day",
    "office_night": "Office Night",
    "tool_room": "Tool Room",
    "utility_room": "Utility Room",
    "utility": "Utility Room",
    "ours": "Ours",
    "fisheyegs": "Fisheye-GS",
    "bathtub": "Bathtub",
    "conference_room": "Conference Room",
    "electrical_room": "Electrical Room",
    "hotel": "Hotel",
    "plant": "Plant",
    "printer": "Printer",
}

def is_smaller(a, b, decimals, include_equal=False):
    if include_equal:
        return round(a, decimals) <= round(b, decimals)
    else:
        return round(a, decimals) < round(b, decimals)

def is_greater(a, b, decimals, include_equal=False):
    if include_equal:
        return round(a, decimals) >= round(b, decimals)
    else:
        return round(a, decimals) > round(b, decimals)

def is_smallest(a, Bs, decimals, include_equal=False):
    return all([is_smaller(a, b, decimals, include_equal=include_equal) for b in Bs])

def is_greatest(a, Bs, decimals, include_equal=False):
    return all([is_greater(a, b, decimals, include_equal=include_equal) for b in Bs])

def compute_normalized_difference(df, metric):
    scenes = sorted(list(set(df_scannet["scene"])))

    relative_scores_ours = np.empty((len(scenes),))
    relative_scores_fisheyegs = np.empty((len(scenes),))

    for idx, scene in enumerate(scenes):
        score_ours = df.loc[(df['scene'] == scene) & (df['method'] == "ours"), metric].values[0]
        score_fisheyegs = df.loc[(df['scene'] == scene) & (df['method'] == "fisheyegs"), metric].values[0]

        relative_scores_ours[idx] = (score_ours - score_fisheyegs) / (0.5 * (score_ours + score_fisheyegs))
        relative_scores_fisheyegs[idx] = (score_fisheyegs - score_ours) / (0.5 * (score_ours + score_fisheyegs))


    mean_relative_score_ours = relative_scores_ours.mean()
    mean_relative_score_fisheyegs = relative_scores_fisheyegs.mean()

    return mean_relative_score_ours, mean_relative_score_fisheyegs

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    # Blender experiments
    csv = os.path.join(args.results_dir, "blender.csv") # "./results/blender.csv"
    df_blender = pd.read_csv(csv)

    # Sort by scene name
    df_blender = df_blender.sort_values(by="scene")

    # Convert to latex string
    table_string = "\\toprule\n"
    table_string += "\\textbf{Scene} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"
    table_string += "\\midrule\n"
    for idx, row in df_blender.iterrows():
        table_string += f"{pretty_names[row['scene']]} & {int(row['gaussians']):,} & {row['psnr']:.{decimals_psnr}f} & {row['ssim']:.{decimals_ssim}f} & {row['lpips']:.{decimals_lpips}f} \\\\\n"
    table_string += "\\bottomrule\n"

    print("######################")
    print("BLENDER")
    print()

    print(table_string)

    # Scannet experiments
    csv = os.path.join(args.results_dir, "scannet.csv") # "./results/scannet.csv"
    df_scannet = pd.read_csv(csv)

    # Sort by (scene, method)
    df_scannet = df_scannet.sort_values(by=["scene", "method"])

    # Convert to latex string
    table_string = "\\toprule\n"
    table_string += "\\textbf{Scene} & \\textbf{Method} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"
    table_string += "\\midrule\n"
    for idx in range(0, len(df_scannet), 2):
        row_fisheyegs = df_scannet.iloc[idx]
        row_ours = df_scannet.iloc[idx+1]

        str_gaussians = f"{int(row_fisheyegs['gaussians']):,}" 
        str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_fisheyegs['gaussians'] < row_ours['gaussians'] else str_gaussians

        str_psnr = f"{row_fisheyegs['psnr']:.{decimals_psnr}f}"
        str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_fisheyegs['psnr'], row_ours['psnr'], decimals_psnr) else str_psnr

        str_ssim = f"{row_fisheyegs['ssim']:.{decimals_ssim}f}"
        str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_fisheyegs['ssim'], row_ours['ssim'], decimals_ssim) else str_ssim

        str_lpips = f"{row_fisheyegs['lpips']:.{decimals_lpips}f}"
        str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_fisheyegs['lpips'], row_ours['lpips'], decimals_lpips) else str_lpips

        table_string += f"{pretty_names[row_fisheyegs['scene']]} & {pretty_names[row_fisheyegs['method']]} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

        str_gaussians = f"{int(row_ours['gaussians']):,}"
        str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_ours['gaussians'] < row_fisheyegs['gaussians'] else str_gaussians

        str_psnr = f"{row_ours['psnr']:.{decimals_psnr}f}"
        str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_ours['psnr'], row_fisheyegs['psnr'], decimals_psnr) else str_psnr

        str_ssim = f"{row_ours['ssim']:.{decimals_ssim}f}"
        str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_ours['ssim'], row_fisheyegs['ssim'], decimals_ssim) else str_ssim

        str_lpips = f"{row_ours['lpips']:.{decimals_lpips}f}"
        str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_ours['lpips'], row_fisheyegs['lpips'], decimals_lpips) else str_lpips

        table_string += f" & {pretty_names[row_ours['method']]} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

        table_string += "\\midrule\n"

    table_string += "\\midrule\n"

    gaussians_ours, gaussians_fisheyegs = compute_normalized_difference(df_scannet, "gaussians")
    psnr_ours, psnr_fisheyegs = compute_normalized_difference(df_scannet, "psnr")
    ssim_ours, ssim_fisheyegs = compute_normalized_difference(df_scannet, "ssim")
    lpips_ours, lpips_fisheyegs = compute_normalized_difference(df_scannet, "lpips")

    phantom = "\\phantom{{-}}"

    str_gaussians_ours = f"{phantom if gaussians_ours >= 0 else ''}{100*gaussians_ours:.2f}\\%"
    str_gaussians_ours = f"\\textbf{{{str_gaussians_ours}}}" if is_smaller(gaussians_ours, gaussians_fisheyegs, 4) else str_gaussians_ours
    str_gaussians_fisheyegs = f"{phantom if gaussians_fisheyegs >= 0 else ''}{100*gaussians_fisheyegs:.2f}\\%"
    str_gaussians_fisheyegs = f"\\textbf{{{str_gaussians_fisheyegs}}}" if is_smaller(gaussians_fisheyegs, gaussians_ours, 2) else str_gaussians_fisheyegs

    str_psnr_ours = f"{phantom if psnr_ours >= 0 else ''}{100*psnr_ours:.2f}\\%"
    str_psnr_ours = f"\\textbf{{{str_psnr_ours}}}" if is_greater(psnr_ours, psnr_fisheyegs, 4) else str_psnr_ours
    str_psnr_fisheyegs = f"{phantom if gaussians_ours >= 0 else ''}{100*psnr_fisheyegs:.2f}\\%"
    str_psnr_fisheyegs = f"\\textbf{{{str_psnr_fisheyegs}}}" if is_greater(psnr_fisheyegs, psnr_ours, 4) else str_psnr_fisheyegs

    str_ssim_ours = f"{phantom if ssim_ours >= 0 else ''}{100*ssim_ours:.2f}\\%"
    str_ssim_ours = f"\\textbf{{{str_ssim_ours}}}" if is_greater(ssim_ours, ssim_fisheyegs, 4) else str_ssim_ours
    str_ssim_fisheyegs = f"{phantom if ssim_fisheyegs >= 0 else ''}{100*ssim_fisheyegs:.2f}\\%"
    str_ssim_fisheyegs = f"\\textbf{{{str_ssim_fisheyegs}}}" if is_greater(ssim_fisheyegs, ssim_ours, 4) else str_ssim_fisheyegs

    str_lpips_ours = f"{phantom if lpips_ours >= 0 else ''}{100*lpips_ours:.2f}\\%"
    str_lpips_ours = f"\\textbf{{{str_lpips_ours}}}" if is_smaller(lpips_ours, lpips_fisheyegs, 4) else str_lpips_ours
    str_lpips_fisheyegs = f"{phantom if lpips_fisheyegs >= 0 else ''}{100*lpips_fisheyegs:.2f}\\%"
    str_lpips_fisheyegs = f"\\textbf{{{str_lpips_fisheyegs}}}" if is_smaller(lpips_fisheyegs, lpips_ours, 4) else str_lpips_fisheyegs


    table_string += f"Relative & {pretty_names['fisheyegs']} & {str_gaussians_fisheyegs} & {str_psnr_fisheyegs} & {str_ssim_fisheyegs} & {str_lpips_fisheyegs} \\\\\n"
    table_string += f"Mean & {pretty_names['ours']} & {str_gaussians_ours} & {str_psnr_ours} & {str_ssim_ours} & {str_lpips_ours} \\\\\n"
        

    table_string += "\\bottomrule\n"

    print("######################")
    print("SCANNET")
    print()

    print(table_string)


    # Jacobian experiments
    csv = os.path.join(args.results_dir, "jacobian.csv") # "./results/jacobian.csv"
    df_jacobian = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Jacobian} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"
    row_first = df_jacobian.iloc[0]
    row_second = df_jacobian.iloc[1]

    str_gaussians = f"{int(row_first['gaussians']):,}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_first['gaussians'] < row_second['gaussians'] else str_gaussians

    str_psnr = f"{row_first['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_first['psnr'], row_second['psnr'], decimals_psnr) else str_psnr

    str_jacobian = "Enabled" if row_first['jacobian'] else "Disabled"
    
    str_ssim = f"{row_first['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_first['ssim'], row_second['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_first['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_first['lpips'], row_second['lpips'], decimals_lpips) else str_lpips

    table_str += f"{pretty_names[row_first['scene']]} & {str_jacobian} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    str_gaussians = f"{int(row_second['gaussians']):,}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_second['gaussians'] < row_first['gaussians'] else str_gaussians

    str_psnr = f"{row_second['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_second['psnr'], row_first['psnr'], decimals_psnr) else str_psnr

    str_jacobian = "Enabled" if row_second['jacobian'] else "Disabled"

    str_ssim = f"{row_second['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_second['ssim'], row_first['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_second['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_second['lpips'], row_first['lpips'], decimals_lpips) else str_lpips

    table_str += f" & {str_jacobian} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    table_str += "\\bottomrule\n"


    print("######################")
    print("JACOBIAN")
    print()

    print(table_str)


    # Skybox experiments
    csv = os.path.join(args.results_dir, "skybox.csv") # 
    df_skybox = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Skybox} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"
    row_first = df_skybox.iloc[0]
    row_second = df_skybox.iloc[1]
    
    str_gaussians = f"{int(row_first['gaussians']):,}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_first['gaussians'] < row_second['gaussians'] else str_gaussians

    str_psnr = f"{row_first['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_first['psnr'], row_second['psnr'], decimals_psnr) else str_psnr

    str_skybox = "Enabled" if row_first['skybox'] else "Disabled"
    
    str_ssim = f"{row_first['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_first['ssim'], row_second['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_first['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_first['lpips'], row_second['lpips'], decimals_lpips) else str_lpips

    table_str += f"{pretty_names[row_first['scene']]} & {str_skybox} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    str_gaussians = f"{int(row_second['gaussians']):,}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_second['gaussians'] < row_first['gaussians'] else str_gaussians

    str_psnr = f"{row_second['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_second['psnr'], row_first['psnr'], decimals_psnr) else str_psnr

    str_skybox = "Enabled" if row_second['skybox'] else "Disabled"

    str_ssim = f"{row_second['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_second['ssim'], row_first['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_second['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_second['lpips'], row_first['lpips'], decimals_lpips) else str_lpips

    table_str += f" & {str_skybox} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    table_str += "\\bottomrule\n"

    print("######################")
    print("SKYBOX")
    print()

    print(table_str)


    # Polydegree experiments
    csv = os.path.join(args.results_dir, "polydegree.csv") # "./results/polydegree.csv"
    df_polydegree = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Coefficients} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"

    for idx in range(0, len(df_polydegree)):
        row = df_polydegree.iloc[idx]

        gaussians_others = [df_polydegree.iloc[i]['gaussians'] for i in range(len(df_polydegree)) if i != idx]
        str_gaussians = f"{int(row['gaussians']):,}"
        str_gaussians = f"\\textbf{{{str_gaussians}}}" if is_smallest(row['gaussians'], gaussians_others, decimals_psnr, include_equal=True) else str_gaussians

        psnr_others = [df_polydegree.iloc[i]['psnr'] for i in range(len(df_polydegree)) if i != idx]
        str_psnr = f"{row['psnr']:.{decimals_psnr}f}"
        str_psnr = f"\\textbf{{{str_psnr}}}" if is_greatest(row['psnr'], psnr_others, decimals_psnr, include_equal=True) else str_psnr

        ssim_others = [df_polydegree.iloc[i]['ssim'] for i in range(len(df_polydegree)) if i != idx]
        str_ssim = f"{row['ssim']:.{decimals_ssim}f}"
        str_ssim = f"\\textbf{{{str_ssim}}}" if is_greatest(row['ssim'], ssim_others, decimals_ssim, include_equal=True) else str_ssim

        lpips_others = [df_polydegree.iloc[i]['lpips'] for i in range(len(df_polydegree)) if i != idx]
        str_lpips = f"{row['lpips']:.{decimals_lpips}f}"
        str_lpips = f"\\textbf{{{str_lpips}}}" if is_smallest(row['lpips'], lpips_others, decimals_lpips, include_equal=True) else str_lpips

        scene_str = pretty_names[row['scene']] if idx == 0 else ""

        table_str += f"{scene_str} & {row['polydegree']} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    table_str += "\\bottomrule\n"

    print("######################")
    print("POLYDEGREE")
    print()

    print(table_str)


    # Latency experiments
    csv = os.path.join(args.results_dir, "latency.csv") # "./results/latency.csv"
    df_latency = pd.read_csv(csv)

    mean_mean = df_latency['mean'].mean()
    mean_std = df_latency['std'].mean()

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Latency / ms} & \\textbf{\#Gaussians} \\\\\n"

    table_str += "\\midrule\n"

    for idx in range(len(df_latency)):
        row = df_latency.iloc[idx]

        str_mean = f"{row['mean']*1e3:.1f}"

        str_std = f"{row['std']*1e3:.1f}"

        scene_str = pretty_names[row['scene']]

        # Get number of gaussians from df_scannet (scene=scene, method=ours)
        gaussians = df_scannet.loc[(df_scannet['scene'] == row['scene']) & (df_scannet['method'] == "ours"), 'gaussians'].values[0]
        gaussians_str = f"{int(gaussians):,}"



        table_str += f"{scene_str} & ${str_mean}\\pm {str_std}$ & {gaussians_str}\\\\\n"

    table_str += "\\bottomrule\n"

    print("######################")
    print("LATENCY")
    print() 

    print(table_str)


    # Scannet Extra experiments
    csv = os.path.join(args.results_dir, "scannet_extra.csv") # "./results/scannet_extra.csv"
    df_scannet_extra = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"

    for idx in range(0, len(df_scannet_extra)):
        row = df_scannet_extra.iloc[idx]

        str_gaussians = f"{int(row['gaussians']):,}"
        str_psnr = f"{row['psnr']:.{decimals_psnr}f}"
        str_ssim = f"{row['ssim']:.{decimals_ssim}f}"
        str_lpips = f"{row['lpips']:.{decimals_lpips}f}"
        scene_str = pretty_names[row['scene']]

        table_str += f"{scene_str} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    table_str += "\\bottomrule\n"

    print("######################")
    print("SCANNET EXTRA")
    print()

    print(table_str)