from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
import json




IMAGE_DIR_PATH = "/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/data/val_images"
SAM_OUTPUT_PATH = "/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/turbo_edit/sam_output_src"
df = pd.read_pickle("/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/data/val.pkl")
json_file = "/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/turbo_edit/dataset/val_turbo.json"

if not os.path.exists(SAM_OUTPUT_PATH):
    os.mkdir(SAM_OUTPUT_PATH)

with open(json_file, "r") as file:
    info = json.load(file)

def save_segmentation_mask(results, output_path):
    mask = results.mask
    mask = mask.astype(np.float64)
    if not mask.any():
        mask = np.zeros((256,256))
    else:
        mask = np.squeeze(mask[-1])
    # mask = np.sum(mask, axis=0)
    mask *= 255.0
    # mask = mask.clip(0, 255).astype(np.uint8)
    # mask = np.squeeze(mask)
    # print(mask.shape)

    img = Image.fromarray(mask)
    
    # print(img)
    if img.mode != "RGB":
        img = img.convert('RGB')
    img.save(output_path)

images = os.listdir(IMAGE_DIR_PATH)

for img, img_info in info.items():
    # target_img_name = img_info["target_image"]
    # tgt_img_idx = int(target_img_name[5:-4])
    src_img_idx = int(img[5:-4])
    img_path = os.path.join(IMAGE_DIR_PATH, img)
    sample = df.iloc[src_img_idx-1]
    category = sample["category"].lower()
    ontology = CaptionOntology(
        {
            category:category
        }
    )
    base_model = GroundedSAM(ontology=ontology)
    results = base_model.predict(img_path)
    # print(results)
    output_img_name = f"mask{src_img_idx}.jpg"
    output_path = os.path.join(SAM_OUTPUT_PATH, output_img_name)
    save_segmentation_mask(results, output_path)





