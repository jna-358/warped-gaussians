import pandas as pd

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

if __name__ == "__main__":
    # Blender experiments
    csv = "./results/blender.csv"
    df_blender = pd.read_csv(csv)

    # Sort by scene name
    df_blender = df_blender.sort_values(by="scene")

    # Convert to latex string
    table_string = "\\toprule\n"
    table_string += "\\textbf{Scene} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"
    table_string += "\\midrule\n"
    for idx, row in df_blender.iterrows():
        table_string += f"{pretty_names[row['scene']]} & {int(row['gaussians'])} & {row['psnr']:.{decimals_psnr}f} & {row['ssim']:.{decimals_ssim}f} & {row['lpips']:.{decimals_lpips}f} \\\\\n"
    table_string += "\\bottomrule\n"

    print("######################")
    print("BLENDER")
    print()

    print(table_string)

    # Scannet experiments
    csv = "./results/scannet.csv"
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

        str_gaussians = f"{int(row_fisheyegs['gaussians'])}" 
        str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_fisheyegs['gaussians'] < row_ours['gaussians'] else str_gaussians

        str_psnr = f"{row_fisheyegs['psnr']:.{decimals_psnr}f}"
        str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_fisheyegs['psnr'], row_ours['psnr'], decimals_psnr) else str_psnr

        str_ssim = f"{row_fisheyegs['ssim']:.{decimals_ssim}f}"
        str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_fisheyegs['ssim'], row_ours['ssim'], decimals_ssim) else str_ssim

        str_lpips = f"{row_fisheyegs['lpips']:.{decimals_lpips}f}"
        str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_fisheyegs['lpips'], row_ours['lpips'], decimals_lpips) else str_lpips

        table_string += f"{pretty_names[row_fisheyegs['scene']]} & {pretty_names[row_fisheyegs['method']]} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

        str_gaussians = f"{int(row_ours['gaussians'])}"
        str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_ours['gaussians'] < row_fisheyegs['gaussians'] else str_gaussians

        str_psnr = f"{row_ours['psnr']:.{decimals_psnr}f}"
        str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_ours['psnr'], row_fisheyegs['psnr'], decimals_psnr) else str_psnr

        str_ssim = f"{row_ours['ssim']:.{decimals_ssim}f}"
        str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_ours['ssim'], row_fisheyegs['ssim'], decimals_ssim) else str_ssim

        str_lpips = f"{row_ours['lpips']:.{decimals_lpips}f}"
        str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_ours['lpips'], row_fisheyegs['lpips'], decimals_lpips) else str_lpips

        table_string += f" & {pretty_names[row_ours['method']]} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

        if idx < len(df_scannet) - 2:
            table_string += "\\midrule\n"
        

    table_string += "\\bottomrule\n"

    print("######################")
    print("SCANNET")
    print()

    print(table_string)


    # Jacobian experiments
    csv = "./results/jacobian.csv"
    df_jacobian = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Jacobian} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"
    row_first = df_jacobian.iloc[0]
    row_second = df_jacobian.iloc[1]

    str_gaussians = f"{int(row_first['gaussians'])}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_first['gaussians'] < row_second['gaussians'] else str_gaussians

    str_psnr = f"{row_first['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_first['psnr'], row_second['psnr'], decimals_psnr) else str_psnr

    str_jacobian = "Enabled" if row_first['jacobian'] else "Disabled"
    
    str_ssim = f"{row_first['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_first['ssim'], row_second['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_first['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_first['lpips'], row_second['lpips'], decimals_lpips) else str_lpips

    table_str += f"{pretty_names[row_first['scene']]} & {str_jacobian} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    str_gaussians = f"{int(row_second['gaussians'])}"
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
    csv = "./results/skybox.csv"
    df_skybox = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Skybox} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"
    row_first = df_skybox.iloc[0]
    row_second = df_skybox.iloc[1]
    
    str_gaussians = f"{int(row_first['gaussians'])}"
    str_gaussians = f"\\textbf{{{str_gaussians}}}" if row_first['gaussians'] < row_second['gaussians'] else str_gaussians

    str_psnr = f"{row_first['psnr']:.{decimals_psnr}f}"
    str_psnr = f"\\textbf{{{str_psnr}}}" if is_greater(row_first['psnr'], row_second['psnr'], decimals_psnr) else str_psnr

    str_skybox = "Enabled" if row_first['skybox'] else "Disabled"
    
    str_ssim = f"{row_first['ssim']:.{decimals_ssim}f}"
    str_ssim = f"\\textbf{{{str_ssim}}}" if is_greater(row_first['ssim'], row_second['ssim'], decimals_ssim) else str_ssim

    str_lpips = f"{row_first['lpips']:.{decimals_lpips}f}"
    str_lpips = f"\\textbf{{{str_lpips}}}" if is_smaller(row_first['lpips'], row_second['lpips'], decimals_lpips) else str_lpips

    table_str += f"{pretty_names[row_first['scene']]} & {str_skybox} & {str_gaussians} & {str_psnr} & {str_ssim} & {str_lpips} \\\\\n"

    str_gaussians = f"{int(row_second['gaussians'])}"
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
    csv = "./results/polydegree.csv"
    df_polydegree = pd.read_csv(csv)

    table_str = "\\toprule\n"
    table_str += "\\textbf{Scene} & \\textbf{Coefficients} & \\textbf{\#Gaussians}$\\downarrow$  & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ \\\\\n"

    table_str += "\\midrule\n"

    for idx in range(0, len(df_polydegree)):
        row = df_polydegree.iloc[idx]

        gaussians_others = [df_polydegree.iloc[i]['gaussians'] for i in range(len(df_polydegree)) if i != idx]
        str_gaussians = f"{int(row['gaussians'])}"
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
