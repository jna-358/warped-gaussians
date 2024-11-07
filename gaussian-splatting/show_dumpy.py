import torch

dump_file = "snapshot_fw.dump"

data = torch.load(dump_file, weights_only=True)

num_elems = len(data)
names = [
    "raster_settings.bg",
    "means3D", 
    "radii", 
    "colors_precomp", 
    "scales", 
    "rotations", 
    "raster_settings.scale_modifier", 
    "cov3Ds_precomp", 
    "raster_settings.viewmatrix", 
    "raster_settings.projmatrix", 
    "raster_settings.tanfovx", 
    "raster_settings.tanfovy", 
    "grad_out_color", 
    "sh", 
    "raster_settings.sh_degree", 
    "raster_settings.campos",
    "geomBuffer",
    "num_rendered",
    "binningBuffer",
    "imgBuffer",
    "raster_settings.debug"
]

props = ["dtype", "shape", "min()", "max()", "requires_grad", "grad_fn", "grad", "is_leaf", "device"]
col_width = 31

header = "name".ljust(col_width)
for prop in props:
    header += prop.ljust(col_width)
print(header)

for i in range(num_elems):
    line = names[i].ljust(col_width)
    for prop in props:
        try:
            val = str(eval(f"data[{i}].{prop}"))
        except:
            val = "N/A"
        line += val.ljust(col_width)
    print(line)