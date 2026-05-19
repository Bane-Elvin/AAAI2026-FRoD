import os
from itertools import product

# Parameter values used to generate the shell scripts
s_values = [0, 0.01, 0.02, 0.1]
s_lr_values = [0, 1e-5, 5e-5, 1e-4, 5e-4]
l_lr_values = [0, 1e-4, 5e-4, 1e-3, 5e-3]

datasets = [
    "/your/path/to/datasets/stanford_cars",
    "/your/path/to/datasets/dtd",
    "/your/path/to/datasets/resisc45",
    "/your/path/to/datasets/sun397"
]


output_dir = "./"
os.makedirs(output_dir, exist_ok=True)


for s, s_lr, l_lr in product(s_values, s_lr_values, l_lr_values):

    if (s == 0 and s_lr != 0) or (s != 0 and s_lr == 0):
        continue

    if l_lr == 0 and not (s in [0.01, 0.02, 0.1] and s_lr in [1e-5, 5e-5, 1e-4, 5e-4]):
        continue


    def format_float(val):
        if val == 0:
            return "0"
        return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+00", "e").replace("e-00", "e-")

    s_str = format_float(s)
    s_lr_str = format_float(s_lr)
    l_lr_str = format_float(l_lr)

    filename = f"train_{s_str}_{s_lr_str}_{l_lr_str}.sh"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        for dataset in datasets:
            f.write(
                f'python clip_vit_peft.py --dataset "{dataset}" --s {s} --s_lr {s_lr} --l_lr {l_lr}\n'
            )

print(f"✅ All scripts have been saved to: {output_dir}")
