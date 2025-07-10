import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from py360convert import (
    e2p,
)  # Importez la fonction de conversion equirectangulaire vers perspective

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    if any(c > 0 for c in crop_factor):
        logging.warning(
            "Crop factors are not directly applied by py360convert. They will be ignored."
        )

    fov_h = 120  # Champ de vision horizontal par défaut pour les projections

    # Initialisation des listes pour les différents types de vues
    horizontal_views = []
    up_views = []
    down_views = []

    # Calcul du nombre de vues pour l'horizon, le haut et le bas
    # On privilégie l'horizon, puis le haut, puis le bas.
    remaining_samples = samples_per_im

    # --- 1. Génération des vues horizontales (pitch = 0) ---
    # Essayons de prendre un minimum de 4 vues horizontales pour une bonne couverture de base,
    # puis distribuons le reste.
    min_horizontal = min(remaining_samples, 4)  # Au moins 4 si possible
    num_horizontal_views = min_horizontal
    if remaining_samples > min_horizontal:
        # Si plus d'échantillons, essayons d'ajouter des vues horizontales jusqu'à ce que la moitié soit atteinte
        num_horizontal_views = max(min_horizontal, (samples_per_im + 1) // 2)
        # On s'assure de ne pas dépasser samples_per_im
        num_horizontal_views = min(num_horizontal_views, remaining_samples)

    if num_horizontal_views > 0:
        yaw_step_horizontal = 360 / num_horizontal_views
        for i in range(num_horizontal_views):
            yaw = i * yaw_step_horizontal
            horizontal_views.append((yaw, 0))  # Pitch 0 pour l'horizon
        remaining_samples -= len(horizontal_views)

    # --- 2. Génération des vues du haut (pitch < 0) ---
    if remaining_samples > 0:
        num_up_views = (
            remaining_samples + 1
        ) // 2  # Moitié restante pour le haut (arrondi vers le haut)
        # S'assurer qu'il y a au moins 1 vue vers le haut si remaining_samples > 0 et samples_per_im > num_horizontal_views
        if (
            num_up_views == 0 and remaining_samples > 0
        ):  # Pour des cas comme samples_per_im = 1
            num_up_views = 1

        pitch_up = -45  # Angle de tangage vers le haut (ajustable)

        if num_up_views > 0:
            yaw_step_up = 360 / num_up_views if num_up_views > 0 else 0
            for i in range(num_up_views):
                yaw = i * yaw_step_up
                up_views.append((yaw, pitch_up))
            remaining_samples -= len(up_views)

    # --- 3. Génération des vues du bas (pitch > 0) ---
    if remaining_samples > 0:
        num_down_views = remaining_samples  # Ce qu'il reste va au bas

        pitch_down = 45  # Angle de tangage vers le bas (ajustable)

        if num_down_views > 0:
            yaw_step_down = 360 / num_down_views if num_down_views > 0 else 0
            for i in range(num_down_views):
                yaw = i * yaw_step_down
                down_views.append((yaw, pitch_down))
            # remaining_samples = 0 ici

    # Combiner les vues dans l'ordre souhaité : horizontal, puis haut, puis bas
    yaw_pitch_pairs = horizontal_views + up_views + down_views

    # Assurez-vous que le nombre total de paires ne dépasse pas samples_per_im et supprimez les doublons
    # (bien que la logique de génération devrait les éviter pour la plupart)
    yaw_pitch_pairs = list(
        dict.fromkeys(yaw_pitch_pairs)
    )  # Conserve l'ordre tout en supprimant les doublons
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
