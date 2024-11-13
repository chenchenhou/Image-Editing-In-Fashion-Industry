from email.mime import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
import json

from sympy import im


def get_parser():
    parser = argparse.ArgumentParser(description="Data Converting")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/train.pkl",
        help="Path to the input pickle file",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        required=True,
        help="Path to the output image directory",
    )
    parser.add_argument(
        "--output_finetune_json_path",
        type=str,
        required=True,
        help="Path to the output json file for finetuning",
    )
    parser.add_argument(
        "--output_image_editing_json_path",
        type=str,
        required=True,
        help="Path to the output json file for image editing",
    )
    return parser


parser = get_parser()
args = parser.parse_args()
output_image_path = args.output_image_path
input_path = args.input_path
output_finetune_json_path = args.output_finetune_json_path
output_image_editing_json_path = args.output_image_editing_json_path

if not os.path.exists(output_image_path):
    os.mkdir(output_image_path)

df = pd.read_pickle(input_path)

data_for_finetuning = []
data_for_image_editing = []

for idx, row in df.iterrows():
    print(f"Processing {idx+1}/{len(df)}")
    target_idx = np.random.randint(0, len(df))
    while target_idx == idx:
        target_idx = np.random.randint(0, len(df))
    src_image_array = row["image"]
    src_description = row["final_description"]

    target_description = df.iloc[target_idx]["description"]
    target_image_array = df.iloc[target_idx]["image"]

    if src_image_array.dtype != np.uint8:
        src_image_array = (src_image_array * 255).astype(np.uint8)

    if target_image_array.dtype != np.uint8:
        target_image_array = (target_image_array * 255).astype(np.uint8)

    src_image = Image.fromarray(src_image_array)
    src_image_filename = f"image{idx+1}.jpg"
    src_image_path = os.path.join(output_image_path, src_image_filename)
    src_image.save(src_image_path, format="JPEG")

    target_image = Image.fromarray(target_image_array)
    target_image_filename = f"image{target_idx+1}.jpg"

    if not os.path.exists(os.path.join(output_image_path, target_image_filename)):
        target_image_path = os.path.join(output_image_path, target_image_filename)
        target_image.save(target_image_path, format="JPEG")

    data_for_finetuning.append(
        {
            "image": src_image_filename,
            "text": src_description,
        }
    )

    data_for_image_editing.append(
        {
            src_image_filename: {
                "src_prompt": src_description,
                "target_image": target_image_filename,
                "tgt_prompt": target_description,
            }
        }
    )

with open(output_finetune_json_path, "w", encoding="utf-8") as f:
    json.dump(data_for_finetuning, f, indent=4)

with open(output_image_editing_json_path, "w", encoding="utf-8") as f:
    json.dump(data_for_image_editing, f, indent=4)
