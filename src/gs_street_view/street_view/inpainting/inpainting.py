import logging
from PIL import Image
from simple_lama_inpainting import SimpleLama

class Inpainter:
    def __init__(self):
        logging.info("Initializing SimpleLama Inpainter...")
        self.simple_lama = SimpleLama()
        logging.info("SimpleLama Inpainter initialized.")

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        logging.info("Performing inpainting...")
        result = self.simple_lama(image, mask)
        logging.info("Inpainting complete.")
        return result
