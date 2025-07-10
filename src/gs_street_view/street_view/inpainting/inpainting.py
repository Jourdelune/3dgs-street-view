import logging
import os

import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


class SimpleLama:
    def __init__(
        self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, images: list[Image.Image], masks: list[Image.Image]) -> list[Image.Image]:
        prepared_images = []
        prepared_masks = []
        for image, mask in zip(images, masks):
            img_tensor, mask_tensor = prepare_img_and_mask(image, mask, self.device)
            prepared_images.append(img_tensor)
            prepared_masks.append(mask_tensor)

        batch_images = torch.cat(prepared_images, dim=0)
        batch_masks = torch.cat(prepared_masks, dim=0)

        with torch.inference_mode():
            inpainted_batch = self.model(batch_images, batch_masks)

            results = []
            for i in range(inpainted_batch.shape[0]):
                cur_res = inpainted_batch[i].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
                results.append(Image.fromarray(cur_res))
            return results


class Inpainter:
    def __init__(self):
        logging.info("Initializing SimpleLama Inpainter...")
        self.simple_lama = SimpleLama()
        logging.info("SimpleLama Inpainter initialized.")

    def inpaint(self, images: list[Image.Image], masks: list[Image.Image]) -> list[Image.Image]:
        logging.info(f"Starting inpainting process for {len(images)} images...")
        results = self.simple_lama(images, masks)
        logging.info("Inpainting complete.")
        return results
