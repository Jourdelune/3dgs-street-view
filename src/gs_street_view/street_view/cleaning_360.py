import logging
import os
import tempfile
from PIL import Image

from gs_street_view.street_view.masking.masking import MaskingProcessor
from gs_street_view.street_view.inpainting.inpainting import Inpainter


class StreetViewCleaner:
    def __init__(self, target_resolution: tuple = None):
        logging.info("Initializing StreetViewCleaner...")
        self.masker = MaskingProcessor()
        self.inpainter = Inpainter()
        self.target_resolution = target_resolution
        logging.info("StreetViewCleaner initialized.")

    def clean_image(self, image_path: str, output_path: str):
        logging.info(f"Cleaning image: {image_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            current_image_path = image_path

            # Aspect ratio correction (moved from masking.py)
            original_image = Image.open(image_path).convert("RGB")
            w, h = original_image.size
            aspect_ratio = w / h
            target_ratio = 2.0
            tolerance = 0.05

            if abs(aspect_ratio - target_ratio) > tolerance:
                raise ValueError(
                    f"Image aspect ratio is {aspect_ratio:.2f}:1, which is not close enough to 2:1. Got {w}x{h}. Please provide an image with a 2:1 aspect ratio or one very close to it."
                )
            elif aspect_ratio != target_ratio:
                new_width = int(target_ratio * h)
                image_for_processing = original_image.resize(
                    (new_width, h), Image.LANCZOS
                )
                temp_aspect_ratio_corrected_path = os.path.join(
                    tmpdir, "temp_aspect_ratio_corrected_image.png"
                )
                image_for_processing.save(temp_aspect_ratio_corrected_path)
                current_image_path = temp_aspect_ratio_corrected_path
                logging.info(
                    f"Image aspect ratio corrected to {new_width}x{h} and saved to temporary path: {current_image_path}"
                )
            else:
                image_for_processing = (
                    original_image  # No aspect ratio correction needed
                )

            if self.target_resolution:
                logging.info(f"Resizing image to {self.target_resolution}...")
                # Use image_for_processing (which might be the original or aspect-ratio corrected)
                resized_image = image_for_processing.resize(
                    self.target_resolution, Image.LANCZOS
                )
                temp_resized_path = os.path.join(tmpdir, "temp_resized_image.png")
                resized_image.save(temp_resized_path)
                current_image_path = temp_resized_path
                logging.info(
                    f"Image resized and saved to temporary path: {current_image_path}"
                )

            # Step 1: Generate mask
            logging.info("Generating mask...")
            # The MaskingProcessor no longer handles aspect ratio correction.
            mask_pil, _ = self.masker.process_image(current_image_path)

            # Save the mask to a temporary file
            temp_mask_path = os.path.join(tmpdir, "temp_mask.png")
            mask_pil.save(temp_mask_path)
            logging.info(f"Mask saved to temporary path: {temp_mask_path}")

            # Load the image for inpainting. Use the current_image_path as the source.
            image_to_inpaint = Image.open(current_image_path).convert("RGB")
            mask_for_inpainting = Image.open(temp_mask_path).convert("L")

            # Step 2: Inpaint the image
            inpainted_image = self.inpainter.inpaint(
                image_to_inpaint, mask_for_inpainting
            )

            # Step 3: Save the inpainted image
            inpainted_image.save(output_path)
            logging.info(f"Inpainted image saved to: {output_path}")

        logging.info(f"Image cleaning process completed for {image_path}")
