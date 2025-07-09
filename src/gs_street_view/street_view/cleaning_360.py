import logging
import os
import tempfile
from PIL import Image

from gs_street_view.street_view.masking.masking import MaskingProcessor
from gs_street_view.street_view.inpainting.inpainting import Inpainter

class StreetViewCleaner:
    def __init__(self):
        logging.info("Initializing StreetViewCleaner...")
        self.masker = MaskingProcessor()
        self.inpainter = Inpainter()
        logging.info("StreetViewCleaner initialized.")

    def clean_image(self, image_path: str, output_path: str):
        logging.info(f"Cleaning image: {image_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Generate mask
            logging.info("Generating mask...")
            # The MaskingProcessor might resize the image and save it to a temporary path.
            # We need to ensure the image used for inpainting is the one that was potentially resized.
            mask_pil, resized_img_path = self.masker.process_image(image_path)
            
            # Save the mask to a temporary file
            temp_mask_path = os.path.join(tmpdir, "temp_mask.png")
            mask_pil.save(temp_mask_path)
            logging.info(f"Mask saved to temporary path: {temp_mask_path}")

            # Load the image for inpainting. Use the resized_img_path if available, otherwise the original.
            image_to_inpaint_path = resized_img_path if resized_img_path else image_path
            image_to_inpaint = Image.open(image_to_inpaint_path).convert("RGB")
            mask_for_inpainting = Image.open(temp_mask_path).convert("L")

            # Step 2: Inpaint the image
            logging.info("Performing inpainting...")
            inpainted_image = self.inpainter.inpaint(image_to_inpaint, mask_for_inpainting)
            logging.info("Inpainting complete.")

            # Step 3: Save the inpainted image
            inpainted_image.save(output_path)
            logging.info(f"Inpainted image saved to: {output_path}")

        logging.info(f"Image cleaning process completed for {image_path}")
