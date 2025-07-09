import cv2
import os
import glob


class MaskAnnotator:
    def __init__(self, image_folder, brush_size=20, display_width=1280):
        self.image_folder = image_folder
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        self.index = 0
        self.brush_size = brush_size
        self.display_width = display_width
        self.drawing = False
        self.history = []

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.draw_mask)

    def draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.history.append(self.mask.copy())
            self._paint_on_mask(x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._paint_on_mask(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self._paint_on_mask(x, y)

    def _paint_on_mask(self, x, y):
        real_x = int(x / self.scale)
        real_y = int(y / self.scale)
        cv2.circle(self.mask, (real_x, real_y), self.brush_size, 255, -1)
        self.mask_display = cv2.resize(
            self.mask,
            (self.display_width, self.img_display.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def run(self):
        while self.index < len(self.image_paths):
            path = self.image_paths[self.index]
            filename = os.path.basename(path)
            mask_path = os.path.splitext(path)[0] + "_mask.png"

            self.img_full = cv2.imread(path)
            self.mask = (
                cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if os.path.exists(mask_path)
                else self._empty_mask(self.img_full)
            )
            self.history = []

            h, w = self.img_full.shape[:2]
            self.scale = self.display_width / w
            self.img_display = cv2.resize(
                self.img_full, (self.display_width, int(h * self.scale))
            )
            self.mask_display = cv2.resize(
                self.mask,
                (self.display_width, int(h * self.scale)),
                interpolation=cv2.INTER_NEAREST,
            )

            print(f"\n[Image {self.index+1}/{len(self.image_paths)}] {filename}")
            print(f"🖌️ Taille brosse : {self.brush_size}")
            self._annotate_image(mask_path)

            self.index += 1

        print("✅ Toutes les images ont été annotées.")

    def _annotate_image(self, mask_path):
        while True:
            vis = self.img_display.copy()
            vis[self.mask_display > 0] = (0, 0, 255)

            cv2.imshow("Image", vis)
            key = cv2.waitKey(20)

            if key == 13:  # Entrée
                cv2.imwrite(mask_path, self.mask)
                print(f"💾 Sauvegardé : {mask_path}")
                break

            elif key == 26:  # Ctrl+Z
                if self.history:
                    self.mask = self.history.pop()
                    self.mask_display = cv2.resize(
                        self.mask,
                        (self.display_width, vis.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    print("↩️ Undo")

            elif key in [43, 82]:  # + ou flèche haut
                self.brush_size += 5
                print(f"➕ Taille brosse : {self.brush_size}")

            elif key in [45, 84]:  # - ou flèche bas
                self.brush_size = max(1, self.brush_size - 5)
                print(f"➖ Taille brosse : {self.brush_size}")

            elif key == 27:  # Échap
                print("👋 Sortie.")
                exit()

    def _empty_mask(self, img):
        return 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 0).astype("uint8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_folder", help="Répertoire contenant les images .jpg à annoter"
    )
    parser.add_argument(
        "--brush", type=int, default=50, help="Taille initiale de la brosse"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Largeur d'affichage max (par défaut: 1280)",
    )
    args = parser.parse_args()

    tool = MaskAnnotator(args.image_folder, args.brush, args.width)
    tool.run()
