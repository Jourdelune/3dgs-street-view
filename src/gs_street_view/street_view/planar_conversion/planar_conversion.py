import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from py360convert import (
    e2p,
)  # Import the equirectangular to perspective conversion function

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _crop_top(bound_arr: list, fov: int, crop_factor: float) -> List[float]:
    """Returns a list of vertical bounds with the bottom cropped."""
    degrees_chopped = 180 * crop_factor
    new_bottom_start = 90 - degrees_chopped - fov / 2
    for i, el in reversed(list(enumerate(bound_arr))):
        if el is not None and el > new_bottom_start + fov / 2:
            bound_arr[i] = None
        elif el is not None and el > new_bottom_start:
            diff = el - new_bottom_start
            bound_arr[i] = new_bottom_start
            for j in range(i - 1, -1, -1):
                if bound_arr[j] is not None:
                    bound_arr[j] -= diff / (2 ** (i - j))
            break
    return bound_arr


def _crop_bottom(bound_arr: list, fov: int, crop_factor: float) -> List[float]:
    """Returns a list of vertical bounds with the top cropped."""
    degrees_chopped = 180 * crop_factor
    new_top_start = -90 + degrees_chopped + fov / 2
    for i, el in enumerate(bound_arr):
        if el is not None and el < new_top_start - fov / 2:
            bound_arr[i] = None
        elif el is not None and el < new_top_start:
            diff = new_top_start - el
            bound_arr[i] = new_top_start
            for j in range(i + 1, len(bound_arr)):
                if bound_arr[j] is not None:
                    bound_arr[j] += diff / (2 ** (j - i))
            break
    return bound_arr


def _crop_bound_arr_vertical(
    bound_arr: list, fov: int, crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
) -> list:
    """Returns a list of vertical bounds adjusted for cropping."""
    if crop_factor[1] > 0:
        bound_arr = _crop_bottom(bound_arr, fov, crop_factor[1])
    if crop_factor[0] > 0:
        bound_arr = _crop_top(bound_arr, fov, crop_factor[0])
    return bound_arr


def compute_resolution_from_equirect(
    image_dir: Path, num_images: int
) -> Tuple[int, int]:
    """Compute the resolution of the perspective projections of equirectangular images
       from the heuristic: num_image * res**2 = orig_height * orig_width.

    Args:
        image_dir: The directory containing the equirectangular images.
    returns:
        The target resolution of the perspective projections.
    """
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".tiff", ".tif"))
    ]
    if not image_files:
        raise ValueError("No images found in the directory.")

    # Just take the first image to determine original resolution
    first_image_path = os.path.join(image_dir, image_files[0])
    with Image.open(first_image_path) as im:
        res_squared = (im.height * im.width) / num_images
        target_side = int(np.sqrt(res_squared))
        return (target_side, target_side)


def generate_planar_projections_from_equirectangular(
    image_dir: Path,
    output_dir: Path,
    planar_image_size: Tuple[int, int],
    samples_per_im: int,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> Path:
    """Generate planar projections from an equirectangular image using py360convert.

    Args:
        image_dir: The directory containing the equirectangular image.
        output_dir: The directory where the planar projections will be saved.
        planar_image_size: The size of the planar projections [width, height].
        samples_per_im: The number of samples to take per image.
        crop_factor: The portion of the image to crop from the (top, bottom, left, and right).
                    Values should be in [0, 1].
    returns:
        The path to the planar projections directory.
    """

    for i in crop_factor:
        if i < 0 or i > 1:
            logging.error("Invalid crop factor. All values must be in [0,1].")
            sys.exit(1)

    yaw_pitch_pairs = []
    fov_h = 120  # Default FOV

    if samples_per_im in [8, 14]:
        left_bound, right_bound = -180, 180
        if crop_factor[3] > 0:
            left_bound = -180 + 360 * crop_factor[3]
        if crop_factor[2] > 0:
            right_bound = 180 - 360 * crop_factor[2]

        if samples_per_im == 8:
            fov_h = 120
            bound_arr = [-45, 0, 45]
            bound_arr = _crop_bound_arr_vertical(bound_arr, fov_h, crop_factor)
            if bound_arr[1] is not None:
                for i in np.arange(left_bound, right_bound, 90):
                    yaw_pitch_pairs.append((i, bound_arr[1]))
            if bound_arr[2] is not None:
                for i in np.arange(left_bound, right_bound, 180):
                    yaw_pitch_pairs.append((i, bound_arr[2]))
            if bound_arr[0] is not None:
                for i in np.arange(left_bound, right_bound, 180):
                    yaw_pitch_pairs.append((i, bound_arr[0]))
        elif samples_per_im == 14:
            fov_h = 110
            bound_arr = [-45, 0, 45]
            bound_arr = _crop_bound_arr_vertical(bound_arr, fov_h, crop_factor)
            if bound_arr[1] is not None:
                for i in np.arange(left_bound, right_bound, 60):
                    yaw_pitch_pairs.append((i, bound_arr[1]))
            if bound_arr[2] is not None:
                for i in np.arange(left_bound, right_bound, 90):
                    yaw_pitch_pairs.append((i, bound_arr[2]))
            if bound_arr[0] is not None:
                for i in np.arange(left_bound, right_bound, 90):
                    yaw_pitch_pairs.append((i, bound_arr[0]))
    else:
        if any(c > 0 for c in crop_factor):
            logging.warning(
                f"samples_per_im={samples_per_im} is not supported for cropping. Crop will be ignored."
            )

        # Initialize lists for different view types
        horizontal_views = []
        up_views = []
        down_views = []

        # Calculate the number of views for the horizon, top, and bottom
        # We prioritize the horizon, then the top, then the bottom.
        remaining_samples = samples_per_im

        # --- 1. Generate horizontal views (pitch = 0) ---
        # Let's try to take a minimum of 4 horizontal views for good basic coverage,
        # then distribute the rest.
        min_horizontal = min(remaining_samples, 4)  # At least 4 if possible
        num_horizontal_views = min_horizontal
        if remaining_samples > min_horizontal:
            # If more samples, let's try to add horizontal views until half is reached
            num_horizontal_views = max(min_horizontal, (samples_per_im + 1) // 2)
            # Make sure not to exceed samples_per_im
            num_horizontal_views = min(num_horizontal_views, remaining_samples)

        if num_horizontal_views > 0:
            yaw_step_horizontal = 360 / num_horizontal_views
            for i in range(num_horizontal_views):
                yaw = i * yaw_step_horizontal
                horizontal_views.append((yaw, 0))  # Pitch 0 for the horizon
            remaining_samples -= len(horizontal_views)

        # --- 2. Generate top views (pitch < 0) ---
        if remaining_samples > 0:
            num_up_views = (
                remaining_samples + 1
            ) // 2  # Remaining half for the top (rounded up)
            # Ensure there is at least 1 upward view if remaining_samples > 0 and samples_per_im > num_horizontal_views
            if (
                num_up_views == 0 and remaining_samples > 0
            ):  # For cases like samples_per_im = 1
                num_up_views = 1

            pitch_up = -45  # Upward pitch angle (adjustable)

            if num_up_views > 0:
                yaw_step_up = 360 / num_up_views if num_up_views > 0 else 0
                for i in range(num_up_views):
                    yaw = i * yaw_step_up
                    up_views.append((yaw, pitch_up))
                remaining_samples -= len(up_views)

        # --- 3. Generate bottom views (pitch > 0) ---
        if remaining_samples > 0:
            num_down_views = remaining_samples  # What's left goes to the bottom

            pitch_down = 45  # Downward pitch angle (adjustable)

            if num_down_views > 0:
                yaw_step_down = 360 / num_down_views if num_down_views > 0 else 0
                for i in range(num_down_views):
                    yaw = i * yaw_step_down
                    down_views.append((yaw, pitch_down))
                # remaining_samples = 0 here

        # Combine the views in the desired order: horizontal, then top, then bottom
        yaw_pitch_pairs = horizontal_views + up_views + down_views

        # Ensure that the total number of pairs does not exceed samples_per_im and remove duplicates
        # (although the generation logic should avoid them for the most part)
        yaw_pitch_pairs = list(
            dict.fromkeys(yaw_pitch_pairs)
        )  # Preserves order while removing duplicates
        if len(yaw_pitch_pairs) > samples_per_im:
            yaw_pitch_pairs = yaw_pitch_pairs[:samples_per_im]

    if not yaw_pitch_pairs:
        logging.error(
            "No valid yaw_pitch_pairs generated. samples_per_im might be too low or logic needs adjustment."
        )
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".tiff", ".tif"))
    ]
    num_ims_total = len(image_files)

    logging.info(f"Processing {num_ims_total} images from {image_dir}")

    for idx, i in enumerate(image_files):
        logging.info(f"Processing image {idx + 1}/{num_ims_total}: {i}")
        image_path = image_dir / i

        pil_image = Image.open(image_path)
        metadata = pil_image.info

        im_np = np.array(pil_image)
        if im_np.ndim == 2:
            im_np = np.stack([im_np] * 3, axis=-1)
        if im_np.shape[2] == 4:
            im_np = im_np[:, :, :3]

        count = 0
        for u_deg, v_deg in yaw_pitch_pairs:
            try:
                pers_image_np = e2p(
                    im_np,
                    fov_h,
                    u_deg,
                    v_deg,
                    out_hw=(planar_image_size[1], planar_image_size[0]),
                    mode="bilinear",
                )

                pers_image_pil = Image.fromarray(pers_image_np)
                output_file_name = f"{Path(i).stem}_{count}.jpg"
                pers_image_pil.save(output_dir / output_file_name, **metadata)
                count += 1
            except Exception as e:
                logging.error(
                    f"Error processing {i} at (yaw={u_deg}, pitch={v_deg}): {e}"
                )
                continue

    logging.info(
        f"Finished generating planar projections. Output saved to {output_dir}"
    )
    return output_dir
