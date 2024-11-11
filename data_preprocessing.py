from ast import arg
from html import parser
from re import sub
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from PIL import Image
import io
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/fashiongen_256_256_train.h5",
        help="Path to the h5 data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output pickle file",
    )
    return parser


parser = get_parser()
args = parser.parse_args()
output_path = args.output_path
file_path = args.input_path

description = []
pose = []
image = []
categories = []
sub_categories = []
gender = []
dataset = h5py.File(file_path, "r")

for i in range(dataset["input_pose"].shape[0]):
    if dataset["input_pose"][i][0] not in [
        b"id_gridfs_3",
        b"id_gridfs_5",
        b"id_gridfs_6",
        b"id_gridfs_2",
    ]:
        gender.append(dataset["input_gender"][i][0])
        categories.append(dataset["input_category"][i][0])
        description.append(dataset["input_description"][i][0])
        pose.append(dataset["input_pose"][i][0])
        sub_categories.append(dataset["input_subcategory"][i][0])
        image.append(dataset["input_image"][i])
print("Last index processed:", i)

description = pd.Series(description)
pose = pd.Series(pose)
sub_categories = pd.Series(sub_categories)
categories = pd.Series(categories)
gender = pd.Series(gender)
image = pd.Series(image, dtype=object)

df = pd.DataFrame()
df["pose"] = pose
df["description"] = description
df["gender"] = gender
df["sub_category"] = sub_categories
df["category"] = categories
df["image"] = image

# Transform each pose to a more human-readable format
df["pose"] = df["pose"].apply(
    lambda x: (
        b"front pose"
        if x == b"id_gridfs_1"
        else (b"full pose" if x == b"id_gridfs_4" else x)
    )
)

# Create a new column with the final description
df["final_description"] = (
    df["pose"]
    + b" of "
    + df["sub_category"]
    + b" for "
    + df["gender"]
    + b". "
    + df["description"]
)

byte_columns = [
    "pose",
    "description",
    "sub_category",
    "category",
    "gender",
    "final_description",
]

for column in byte_columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8", errors="replace").replace("\ufffd", "").lower()
    )

df.to_pickle(output_path)
