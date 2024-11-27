import os
import cv2
from matplotlib import category
import pip
from diffusers import AutoPipelineForImage2Image
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    mask_pil_to_torch,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_timesteps,
    retrieve_latents,
)
import torch
import PIL
import jsonc as json
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForInpainting,
    StableDiffusionXLInpaintPipeline,
)
from functools import partial
from utils import SAMPLING_DEVICE, get_ddpm_inversion_scheduler, create_xts, device
from config import get_config
import argparse
from diffusers.utils import load_image
from torchvision.transforms.functional import to_pil_image
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import shutil
import pandas as pd
import time


VAE_SAMPLE = "argmax"  # "argmax" or "sample"
RESIZE_TYPE = None  # Image.LANCZOS


def encode_image(image, pipe, generator):
    pipe_dtype = pipe.dtype
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=device, dtype=pipe.dtype)

    if pipe.vae.config.force_upcast:
        image = image.float()
        pipe.vae.to(dtype=torch.float32)

    init_latents = retrieve_latents(
        pipe.vae.encode(image), generator=generator, sample_mode=VAE_SAMPLE
    )

    if pipe.vae.config.force_upcast:
        pipe.vae.to(pipe_dtype)

    init_latents = init_latents.to(pipe_dtype)
    init_latents = pipe.vae.config.scaling_factor * init_latents

    return init_latents


def prepare_mask(
    mask: Union[PIL.Image.Image, np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Prepares a mask to be consumed by the Stable Diffusion pipeline. This means that this input will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``1`` for
    the ``mask``.

    The ``mask`` will be binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        torch.Tensor: mask as ``torch.Tensor`` with 4 dimensions: ``batch x channels x height x width``.
    """
    if isinstance(mask, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not"
            )

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
            )
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # mask = torch.from_numpy(mask)

    return mask


def blur_mask(
    mask: np.ndarray, blur_kernel_size: int = 15, dilation_iterations: int = 1
) -> torch.Tensor:
    """
    Applies a blur to the edges of the mask with adjustable edge thickness.

    :param mask: Input binary mask as a numpy array (values in [0, 255]).
    :param blur_kernel_size: Size of the Gaussian blur kernel.
    :param dilation_iterations: Number of iterations for edge dilation to control thickness.
    :return: Blurred mask as a torch tensor scaled to [0, 1].
    """
    # Convert mask to binary (0 or 255)
    mask = mask * 255
    binary_mask = (mask > 127).astype(np.uint8) * 255

    # Find and dilate edges
    edges = cv2.Canny(binary_mask, 100, 200)
    dilated_edges = cv2.dilate(edges, None, iterations=dilation_iterations)

    # Create a blurred mask
    blurred_mask = cv2.GaussianBlur(
        binary_mask, (blur_kernel_size, blur_kernel_size), 0
    )

    # Combine blurred edges with the original mask
    result_mask = np.where(dilated_edges > 0, blurred_mask, binary_mask)

    # Scale result back to [0, 1] and convert to torch tensor
    result_mask = result_mask.astype(np.float32) / 255.0
    result_mask = torch.tensor(result_mask, dtype=torch.float32)

    return result_mask


def set_pipeline(pipeline: StableDiffusionXLImg2ImgPipeline, num_timesteps, generator):
    if num_timesteps == 3:
        config_from_file = "run_configs/noise_shift_steps_3.yaml"
    elif num_timesteps == 4:
        config_from_file = "run_configs/noise_shift_steps_4.yaml"
    else:
        config_from_file = "run_configs/noise_shift_steps.yaml"

    config = get_config(config_from_file)
    if config.timesteps is None:

        denoising_start = config.step_start / config.num_steps_inversion  # 1/5
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, config.num_steps_inversion, device, None
        )  # tensor([999, 799, 599, 399, 199], device='cuda:0')
        timesteps, num_inference_steps = pipeline.get_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            denoising_start=denoising_start,
            strength=0,
        )  # tensor([799, 599, 399, 199], device='cuda:0')
        timesteps = timesteps.type(torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            num_inference_steps=config.num_steps_inversion,
            guidance_scale=0,
            generator=generator,
            denoising_start=denoising_start,
            strength=0,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=timesteps.cpu(),
        )
    else:
        timesteps = torch.tensor(config.timesteps, dtype=torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            timesteps=timesteps,
            guidance_scale=0,
            denoising_start=0,
            strength=1,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=config.timesteps,  # device=pipeline.device
        )
    timesteps = [torch.tensor(t) for t in timesteps.tolist()]
    return timesteps, config


def run(
    image_path,
    src_prompt,
    tgt_prompt,
    seed,
    w1,
    num_timesteps,
    pipeline: StableDiffusionXLImg2ImgPipeline,
    mask_path,
    blur = True,
    blur_kernel_size = 15,
    dilation_iterations = 1,
):

    generator = torch.Generator(device=SAMPLING_DEVICE).manual_seed(seed)

    timesteps, config = set_pipeline(pipeline, num_timesteps, generator)

    x_0_image = Image.open(image_path).convert("RGB").resize((512, 512), RESIZE_TYPE)

    image_np = np.array(x_0_image)
    image_height, image_width = image_np.shape[:2]

    # polygons = load_polygons(annotation_path, image_width, image_height)
    # if not polygons:
    #     print(f"No polygons found in {annotation_path}. Skipping.")
    #     return
    # target_class_ids = extract_target_class_ids(tgt_prompt, label_to_class_id)

    # mask_array = create_mask_from_polygons(
    #     (image_width, image_height), polygons, target_class_ids
    # )
    # mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
    # mask_image.save(f"output/mask_{os.path.basename(image_path)}")
    # mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float().to(device)

    # timestpes = [799, 599, 399, 199] in the case of 4 steps
    # x_0 = pipeline.image_processor.preprocess(x_0_image)
    # x_0 = x_0.to(device=pipeline.device, dtype=pipeline.dtype)
    # print(x_0.shape)
    x_0 = encode_image(x_0_image, pipeline, generator)
    x_ts = create_xts(
        config.noise_shift_delta,
        config.noise_timesteps,
        generator,
        pipeline.scheduler,
        timesteps,
        x_0,
    )
    mask_image_raw = (
        Image.open(mask_path).convert("RGB").resize((512, 512), RESIZE_TYPE)
    )
    mask = prepare_mask(mask_image_raw)
    if blur == True:
        mask = blur_mask(mask, blur_kernel_size=blur_kernel_size, dilation_iterations=dilation_iterations)
    else:
        mask = torch.from_numpy(mask)

    height, width = mask.shape[-2:]
    # Make the mask same dimension as latent image
    mask = torch.nn.functional.interpolate(
        mask,
        size=(height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
    ).squeeze(0)
    mask = mask.to(x_0.device)
    mask = mask.to(x_0.dtype)
    x_ts = [xt.to(dtype=x_0.dtype) for xt in x_ts]
    latents = [x_ts[0]]
    pipeline.scheduler = get_ddpm_inversion_scheduler(
        pipeline.scheduler,
        config,
        timesteps,
        latents,
        x_ts,
        mask,
    )
    pipeline.scheduler.w1 = w1

    latent = latents[0].expand(3, -1, -1, -1)
    prompt = [src_prompt, src_prompt, tgt_prompt]
    start = time.time()
    image = pipeline.__call__(image=latent, prompt=prompt).images
    end = time.time()
    return image[2], end - start


def load_pipe(fp16, cache_dir):
    kwargs = (
        {
            "torch_dtype": torch.float16,
            "variant": "fp16",
        }
        if fp16
        else {}
    )
    pipeline: StableDiffusionXLImg2ImgPipeline = (
        AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            safety_checker=None,
            cache_dir=cache_dir,
            **kwargs,
        )
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDPMScheduler.from_pretrained(  # type: ignore
        "stabilityai/sdxl-turbo",
        subfolder="scheduler",
    )

    return pipeline


def load_polygons(annotation_path, image_width, image_height):
    polygons = []
    with open(annotation_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 2:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                # Group coordinates into (x, y) pairs
                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * image_width)
                    y = int(coords[i + 1] * image_height)
                    polygon.append((x, y))
                polygons.append((class_id, polygon))
    return polygons


def create_mask_from_polygons(image_size, polygons, target_class_ids):
    mask = Image.new("L", image_size, 0)  # Create a blank mask image
    draw = ImageDraw.Draw(mask)
    for class_id, polygon in polygons:
        if class_id in target_class_ids:
            draw.polygon(polygon, outline=1, fill=1)
    mask_array = np.array(mask)
    return mask_array


def extract_target_class_ids(target_prompt, label_to_class_id):
    target_class_ids = []
    for label, class_id in label_to_class_id.items():
        if label.lower() in target_prompt.lower():
            target_class_ids.append(class_id)
    return target_class_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/turbo_edit/dataset/dataset.json",
    )
    parser.set_defaults(fp16=False)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--w", type=float, default=1.5)
    parser.add_argument("--timesteps", type=int, default=4)  # 3 or 4
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_pickle", type=str, default="../data/val.pkl")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/home/mmpug/revanth/Image-Editing-In-Fashion-Industry/turbo_edit/sam_output",
    )

    args = parser.parse_args()

    img_paths_to_prompts = json.load(open(args.prompts_file, "r"))
    eval_dataset_folder = "/".join(args.prompts_file.split("/")[:-1]) + "/"
    img_paths = [
        f"{eval_dataset_folder}/{img_name}" for img_name in img_paths_to_prompts.keys()
    ]
    # DATASET_DIR_PATH = "labeled_dataset"
    # if os.path.exists(DATASET_DIR_PATH):
    #     shutil.rmtree(DATASET_DIR_PATH)

    # df = pd.read_pickle(args.data_pickle)
    # categories = set(df["category"].values)
    # cleaned_categories = {}
    # for c in categories:
    #     if "&" in c:
    #         c = c.split("&")[0].strip()
    #     cleaned_categories[c] = c
    # ontology = CaptionOntology(
    #     {
    #         "shirt": "shirt",
    #         "t-shirt": "t-shirt",
    #         "glasses": "glasses",
    #         "pants": "pants",
    #         "hat": "hat",
    #         "dress": "dress",
    #         "skirt": "skirt",
    #         "shoes": "shoes",
    #     }
    # )
    # ontology = CaptionOntology(cleaned_categories)
    # ontology_labels = [label for label, _ in ontology.promptMap]

    # label_to_class_id = {label: idx for idx, label in enumerate(ontology_labels)}

    # base_model = GroundedSAM(ontology)
    # IMAGE_DIR_PATH = os.path.join("dataset")

    # dataset = base_model.label(
    #     input_folder=IMAGE_DIR_PATH,
    #     extension=".jpg",
    #     output_folder=DATASET_DIR_PATH,
    # )

    # IMAGE_DIR_PATH = os.path.join(DATASET_DIR_PATH, "valid", "images")
    # ANNOTATIONS_DIR_PATH = os.path.join(DATASET_DIR_PATH, "valid", "labels")
    # img_paths = [
    #     os.path.join(IMAGE_DIR_PATH, img_name)
    #     for img_name in img_paths_to_prompts.keys()
    # ]

    pipeline = load_pipe(args.fp16, args.cache_dir)
    running_times = 0.0
    for i, img_path in enumerate(img_paths):
        img_name = img_path.split("/")[-1]
        prompt = img_paths_to_prompts[img_name]["src_prompt"]
        edit_prompts = img_paths_to_prompts[img_name]["tgt_prompt"]
        tgt_image_name = img_paths_to_prompts[img_name]["target_image"]
        mask_path = os.path.join(args.mask_dir, tgt_image_name.replace("image", "mask"))

        # annotation_name = img_name.replace(".jpg", ".txt")
        # annotation_path = os.path.join(ANNOTATIONS_DIR_PATH, annotation_name)

        res, diffusion_time = run(
            img_path,
            prompt,
            edit_prompts[0],
            args.seed,
            args.w,
            args.timesteps,
            pipeline=pipeline,
            mask_path=mask_path,
        )
        running_times += diffusion_time
        os.makedirs(args.output_dir, exist_ok=True)
        res.save(f"{args.output_dir}/output_{i}.png")

    print(f"Total time: {running_times}")
    print(f"Average time: {running_times / len(img_paths)}")
