import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import SamModel, SamProcessor
from torch.cuda.amp import autocast


def get_bounding_box(ground_truth_map):
    """Returns a bounding box for the entire map."""
    return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]


def resize_mask(mask, size):
    """
    Resizes a mask tensor.
    mask : tensor shape (batch_size, 1, H, W)
    size : tuple (new_height, new_width)
    """
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False)


class MaskingProcessor:
    def __init__(self, model_name: str = "Jour/sam-vit-base-equirectangular-finetuned"):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path: str):
        """
        Takes a 360 equirectangular image, decomposes it into two square patches,
        generates a mask for each, merges them, smoothes the result, and
        resizes it to the original image size.

        Args:
            image_path (str): Path to the 360 image.

        Returns:
            Image.Image: The final processed mask as a PIL Image.
            str: Path to the resized image if it was resized, otherwise the original image path.
        """
        # Load image
        original_image = Image.open(image_path).convert("RGB")
        w, h = original_image.size

        # Check aspect ratio
        aspect_ratio = w / h
        target_ratio = 2.0
        tolerance = 0.05  # 5% tolerance

        resized_image_path = image_path
        image = original_image  # Initialize image to original_image

        if abs(aspect_ratio - target_ratio) > tolerance:
            raise ValueError(
                f"Image aspect ratio is {aspect_ratio:.2f}:1, which is not close enough to 2:1. Got {w}x{h}. Please provide an image with a 2:1 aspect ratio or one very close to it."
            )
        elif aspect_ratio != target_ratio:
            # Resize image to exactly 2:1 aspect ratio
            new_width = int(target_ratio * h)
            image = original_image.resize((new_width, h), Image.LANCZOS)

            # Save the resized image
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            resized_image_path = os.path.join(
                os.path.dirname(image_path), f"resized_{name}{ext}"
            )
            image.save(resized_image_path)
            logging.info(
                f"Image resized to {new_width}x{h} and saved to {resized_image_path}"
            )

        original_w, original_h = original_image.size  # Store original dimensions

        # Split image into two halves (cubes) and resize to 256x256
        left_half = image.crop((0, 0, h, h)).resize((256, 256), Image.LANCZOS)
        right_half = image.crop((h, 0, w, h)).resize(
            (256, 256), Image.LANCZOS
        )  # Use w from potentially resized image

        halves = [left_half, right_half]
        processed_masks = []

        with torch.no_grad():
            for half_img in halves:
                # Create a dummy ground truth map to get a bounding box for the whole image
                dummy_gt = np.zeros(half_img.size)
                prompt = get_bounding_box(dummy_gt)

                # Prepare image and prompt for the model
                inputs = self.processor(
                    half_img, input_boxes=[[prompt]], return_tensors="pt"
                ).to(self.device)

                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(**inputs, multimask_output=False)

                # Post-process mask
                mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                mask_prob = mask_prob.cpu().numpy().squeeze()
                hard_mask = (mask_prob > 0.5).astype(np.uint8)
                processed_masks.append(hard_mask)

        # Merge the two masks
        merged_mask_np = np.concatenate(processed_masks, axis=1)

        # Smooth the mask
        # Using a gaussian filter for smoothing. The sigma value can be adjusted.
        smoothed_mask_np = gaussian_filter(merged_mask_np.astype(float), sigma=5)

        # Normalize to 0-255 range and convert to image
        smoothed_mask_np = (smoothed_mask_np > 0.5).astype(np.uint8) * 255

        # Resize the mask to the original image size
        final_mask_image = Image.fromarray(smoothed_mask_np).resize(
            image.size, Image.LANCZOS
        )

        return final_mask_image, resized_image_path
