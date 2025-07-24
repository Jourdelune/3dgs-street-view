import logging
import os
import random

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PanoramaxDownloader:
    """
    A class for downloading image sequences from Panoramax.

    This class provides methods to find and download image sequences based on
    various criteria such as location, quality, and collection.
    """

    BASE_API_URL = "https://api.panoramax.xyz/api"

    def __init__(self, download_dir="panoramax_downloads"):
        """
        Initializes the PanoramaxDownloader.

        Args:
            download_dir (str): The directory where images will be downloaded.
                                Defaults to "panoramax_downloads".
        """
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            logging.info(f"Created download directory: {self.download_dir}")

    def _calculate_quality_grade(self, horizontal_accuracy, pixel_density):
        """
        Calculates the quality grade (A, B, C, D, E) based on Panoramax documentation.

        Args:
            horizontal_accuracy (float, optional): The horizontal accuracy of the GPS data.
            pixel_density (float, optional): The horizontal pixel density.

        Returns:
            str: The calculated quality grade.
        """
        gps_score = 1
        if horizontal_accuracy is not None:
            if horizontal_accuracy <= 1.0:
                gps_score = 5
            elif horizontal_accuracy <= 2.0:
                gps_score = 4
            elif horizontal_accuracy <= 5.0:
                gps_score = 3
            elif horizontal_accuracy <= 10.0:
                gps_score = 2

        density_score = 1
        if pixel_density is not None:
            if pixel_density >= 30:
                density_score = 5
            elif pixel_density >= 15:
                density_score = 4
            elif pixel_density < 15:
                density_score = 3

        overall_score = (gps_score * 0.2) + (density_score * 0.8)

        if overall_score >= 4.5:
            grade = "A"
        elif overall_score >= 3.5:
            grade = "B"
        elif overall_score >= 2.5:
            grade = "C"
        elif overall_score >= 1.5:
            grade = "D"
        else:
            grade = "E"
        logging.debug(
            f"Calculated grade: {grade} (GPS: {gps_score}, Density: {density_score}, Overall: {overall_score:.2f})"
        )
        return grade

    def find_collections(self, limit=1000, bbox=None, filters=None) -> dict:
        """
        Finds collections in Panoramax based on specified criteria.

        Args:
            limit (int, optional): Max number of collections to return. Defaults to 1000.
            bbox (list, optional): A list of four coordinates representing the bounding
                                   box [lon_min, lat_min, lon_max, lat_max]. Defaults to None.
            filters (dict, optional): A dictionary of filters to apply to the search. Defaults to None.
        Returns:
            dict: A dictionary where keys are collection IDs and values are collection details.
                   Returns None if no collections are found or if an error occurs.
        """
        search_url = f"{self.BASE_API_URL}/collections"
        params = {"limit": str(limit)}
        if bbox:
            params["bbox"] = ",".join(map(str, bbox))
        if filters:
            params["filter"] = " and ".join([f"{k}={v}" for k, v in filters.items()])

        logging.info(f"Searching for collections with params: {params}")
        try:
            response = requests.get(search_url, params=params)
            logging.info(f"API request URL: {response.url}")
            response.raise_for_status()
            data = response.json()
            logging.info(
                f"Found {len(data.get('collections', []))} collections from API."
            )

            # merge all collections into a single dictionary
            collections = {}
            for collection in data.get("collections", []):
                collection_id = collection.get("id")
                collections[collection_id] = collection
            logging.info(f"Found {len(collections)} collections.")
            return collections

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def find_sequence(
        self,
        bbox=None,
        limit=5000,
        min_photos_in_sequence=10,
        required_quality="D",
        collection=None,
        field_of_view=None,
    ):
        """
        Finds eligible image sequences from Panoramax based on specified criteria.

        Args:
            bbox (list, optional): A list of four coordinates representing the bounding
                                   box [lon_min, lat_min, lon_max, lat_max]. Defaults to None.
            limit (int, optional): The maximum number of features to return from the API.
                                   Defaults to 5000.
            min_photos_in_sequence (int, optional): The minimum number of photos required
                                                    in a sequence to be considered eligible.
                                                    Defaults to 10.
            required_quality (str, optional): The minimum required quality grade for the photos.
                                              Defaults to "B".
            collection (str, optional): The specific collection to search within.
                                        Defaults to None.
            field_of_view (int, optional): The required field of view for the images.
                                           Defaults to None.

        Returns:
            dict: A dictionary where keys are sequence IDs and values are lists of photo
                  dictionaries for eligible sequences. Returns None if no eligible
                  sequences are found or if an error occurs.
        """
        search_url = f"{self.BASE_API_URL}/search"
        params = {"limit": str(limit)}
        if bbox:
            params["bbox"] = ",".join(map(str, bbox))
        if collection:
            params["collection"] = collection

        filters = []
        if field_of_view is not None:
            filters.append(f"field_of_view={field_of_view}")

        if filters:
            params["filter"] = " and ".join(filters)

        logging.info(f"Searching for sequences with params: {params}")
        try:
            response = requests.get(search_url, params=params)
            logging.info(f"API request URL: {response.url}")
            response.raise_for_status()
            data = response.json()
            logging.info(
                f"Found {len(data.get('features', []))} total photos from API."
            )
            photos_by_sequence = {}
            for feature in data.get("features", []):
                pic_id = feature.get("id")
                image_url = feature.get("assets", {}).get("hd", {}).get("href")
                horizontal_accuracy = feature.get("properties", {}).get(
                    "quality:horizontal_accuracy"
                )
                pixel_density = feature.get("properties", {}).get(
                    "panoramax:horizontal_pixel_density"
                )
                sequence_id = feature.get("collection")

                calculated_grade = self._calculate_quality_grade(
                    horizontal_accuracy, pixel_density
                )

                if (
                    pic_id
                    and image_url
                    and sequence_id
                    and calculated_grade <= required_quality
                ):
                    if sequence_id not in photos_by_sequence:
                        photos_by_sequence[sequence_id] = []
                    photos_by_sequence[sequence_id].append(
                        {
                            "id": pic_id,
                            "url": image_url,
                            "grade": calculated_grade,
                            "sequence_id": sequence_id,
                        }
                    )

            eligible_sequences = {
                seq_id: photos
                for seq_id, photos in photos_by_sequence.items()
                if len(photos) >= min_photos_in_sequence
            }

            logging.info(
                f"Found {len(eligible_sequences)} eligible sequences with at least {min_photos_in_sequence} photos."
            )

            if not eligible_sequences:
                logging.warning("No eligible sequences found.")
                return None

            return eligible_sequences

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def download_sequence(self, sequence_id, photos, max_photos=50):
        """
        Downloads photos from a given sequence.

        Args:
            sequence_id (str): The ID of the sequence to download.
            photos (list): A list of photo dictionaries to download.
            max_photos (int, optional): The maximum number of photos to download
                                        from the sequence. Defaults to 50.
        """
        if not photos:
            logging.error("No photos provided to download.")
            return

        pictures_to_download = photos[:max_photos]
        logging.info(
            f"Starting download of {len(pictures_to_download)} photos for sequence {sequence_id}."
        )

        sequence_download_dir = os.path.join(self.download_dir, sequence_id)
        if not os.path.exists(sequence_download_dir):
            os.makedirs(sequence_download_dir)
            logging.info(f"Created directory for sequence: {sequence_download_dir}")

        for i, pic in enumerate(pictures_to_download):
            pic_id = pic["id"]
            image_url = pic["url"]
            file_path = os.path.join(sequence_download_dir, f"{i}_{pic_id}.jpg")

            if os.path.exists(file_path):
                logging.warning(f"File {file_path} already exists. Skipping download.")
                continue

            try:
                logging.info(f"Downloading photo {pic_id} from {image_url}")
                image_response = requests.get(image_url, stream=True)
                image_response.raise_for_status()

                with open(file_path, "wb") as f:
                    for chunk in image_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"Photo {pic_id} saved to {file_path}")
                logging.info(f"Direct link: https://api.panoramax.xyz/?pic={pic['id']}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download {pic_id}: {e}")
                continue


if __name__ == "__main__":
    # This is an example of how to use the PanoramaxDownloader class.
    # It will search for a random sequence in a specific bounding box (Paris)
    # and download the first 5 photos of that sequence.

    downloader = PanoramaxDownloader(download_dir="downloaded_panoramax_sequences")

    # Bounding box for Paris, France
    paris_bbox = [2.224, 48.815, 2.469, 48.902]

    logging.info(f"Searching for sequences in bbox: {paris_bbox}")
    sequences = downloader.find_sequence(
        bbox=paris_bbox, min_photos_in_sequence=10, required_quality="B"
    )

    if sequences:
        # Pick a random sequence to download
        random_sequence_id = random.choice(list(sequences.keys()))
        photos_to_download = sequences[random_sequence_id]

        logging.info(f"Selected random sequence to download: {random_sequence_id}")
        downloader.download_sequence(
            random_sequence_id, photos_to_download, max_photos=5
        )
        logging.info("Example download finished.")
    else:
        logging.warning("No sequences found for the given criteria.")
