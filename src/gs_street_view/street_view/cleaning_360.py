import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

from gs_street_view import (
    Inpainter,
    MaskingProcessor,
    compute_resolution_from_equirect,
    generate_planar_projections_from_equirectangular,
)


class StreetViewCleaner:
    def __init__(self, target_resolution: tuple = None):
        logging.info("Initializing StreetViewCleaner...")
        self.masker = MaskingProcessor()
        self.inpainter = Inpainter()
        self.target_resolution = target_resolution
        logging.info("StreetViewCleaner initialized.")

    def _preprocess_image(self, image_path: str) -> Image.Image:
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
            processed_image = original_image.resize((new_width, h), Image.LANCZOS)
            logging.info(f"Image aspect ratio corrected to {new_width}x{h}")
        else:
            processed_image = original_image

        if self.target_resolution:
            logging.info(f"Resizing image to {self.target_resolution}...")
            processed_image = processed_image.resize(self.target_resolution, Image.LANCZOS)
            logging.info("Image resized")
        return processed_image

    def clean_image(
        self,
        image_path: str,
        output_path: str,
        sample_per_image: int = 14,
        crop_factors: tuple = (0.0, 0.2, 0.0, 0.0),
    ):
        logging.info(f"Cleaning image: {image_path}")

        os.makedirs(output_path, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            
            processed_image = self._preprocess_image(image_path)

            os.makedirs(Path(tmpdir) / "image", exist_ok=True)
            os.makedirs(Path(tmpdir) / "mask", exist_ok=True)

            temp_image_path = Path(tmpdir) / "image" / "temp_image.png"
            processed_image.save(temp_image_path)

            # Step 1: Generate mask
            logging.info("Generating mask...")
            mask_pil, _ = self.masker.process_image(str(temp_image_path))

            temp_mask_path = Path(tmpdir) / "mask" / "temp_mask.png"
            mask_pil.convert("RGB").save(temp_mask_path)
            logging.info(f"Mask saved to temporary path: {temp_mask_path}")

            self._process_planar_projections(
                Path(tmpdir),
                image_path,
                output_path,
                sample_per_image,
                crop_factors,
            )

        logging.info(f"Image cleaning process completed for {image_path}")

    def _process_planar_projections(
        self,
        tmpdir_path: Path,
        original_image_path: str,
        output_path: str,
        sample_per_image: int,
        crop_factors: tuple,
    ):
        logging.info("Generating planar projections from equirectangular image...")
        target_resolution = compute_resolution_from_equirect(
            tmpdir_path / "image", sample_per_image
        )

        output_planar_dir_image = generate_planar_projections_from_equirectangular(
            tmpdir_path / "image",
            tmpdir_path / "image_planar",
            target_resolution,
            sample_per_image,
            crop_factors,
        )

        output_planar_dir_mask = generate_planar_projections_from_equirectangular(
            tmpdir_path / "mask",
            tmpdir_path / "mask_planar",
            target_resolution,
            sample_per_image,
            crop_factors,
        )

        for i, image_filename in enumerate(os.listdir(output_planar_dir_image)):
            mask_filename = image_filename.replace("image", "mask")
            logging.info(f"Processing image: {image_filename} with mask: {mask_filename}")

            image_to_inpaint = Image.open(
                output_planar_dir_image / image_filename
            ).convert("RGB")
            mask_for_inpainting = Image.open(
                output_planar_dir_mask / mask_filename
            ).convert("L")

            inpainted_image = self.inpainter.inpaint(
                image_to_inpaint, mask_for_inpainting
            )

            output_filename = (
                Path(original_image_path).stem + f"_{i}.png"
            )
            inpainted_image.save(os.path.join(output_path, output_filename))
            logging.info(f"Inpainted image saved as: {os.path.join(output_path, output_filename)}")
