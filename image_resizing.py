#image resizing to 224 * 448, input : directory including sceanario
import cv2
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger

def make_parser():
    parser = argparse.ArgumentParser(description="Resize images to 224 x 448")
    parser.add_argument(
        "-i", "--input_dir", default = "../../Dataset/00.scenario_v1/demo_251120", type = str, help = "input directory"
    )
    parser.add_argument(
        "-o", "--output_dir", default = "../../Dataset/00.scenario_v1/resize/demo_251120", type = str, help = "output directory"
    )
    parser.add_argument(
        "--width", default = 448, type = int, help = "output width"
    )
    parser.add_argument(
        "--height", default = 224, type = int, help = "output height"
    )
    parser.add_argument(
        "-method", default = "opencv", choices = ["opencv", ], help = "resize method"
    )
    return parser

class ImageResizer:
    def __init__(self, input_dir, output_dir, target_width = 448, target_height=224, method = "opencv"):
        """
        Args:
            input_dir: 입력 이미지 디렉토리
            output_dir: 출력 이미지 디렉토리  
            target_width: 목표 너비 (기본값: 448)
            target_height: 목표 높이 (기본값: 224)
            method: 리사이징 방법 ("opencv" 또는 "pillow")
        """

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_width = target_width
        self.target_height = target_height
        self.method = method

        self.image_extensions = {".png"}

        self.output_dir.mkdir(parents = True, exist_ok = True)

    def resize_image_opencv(self, image_path, output_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to load image: {image_path}")
                return False

            resized_img = cv2.resize(img, (self.target_width, self.target_height))

            cv2.imwrite(str(output_path), resized_img)
            return True
        except Exception as e:
            print(f"Error resizing {image_path} w/ Opencv: {e}")
            return False
    def resize_single_image(self, image_path, output_path):
        if self.method == "opencv":
            return self.resize_image_opencv(image_path, output_path)





    def get_image_files(self):
        image_files = []
        image_files.extend(self.input_dir.rglob("*.png"))
        image_files.extend(self.input_dir.rglob("*.PNG"))
        return image_files
    
    def process_all_images(self):
        image_files = self.get_image_files()

        if not image_files:
            print("No PNG files found in the input directory")
            return
        
        print(f"Found {len(image_files)} PNG files to process")
        print("-" * 50)

        success_count = 0

        for i, image_path in enumerate(image_files, 1):
            relative_path = image_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path

            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing {i}/{len(image_files)}")

            #resizeing image
            if self.resize_single_image(image_path, output_path):
                success_count += 1
                print(f"  ✓ Resized successfully -> {relative_path}")
            else:
                print(f"  ✗ Failed to resize -> {relative_path}")
        
        print("-" * 50)
        print(f"Processing complete!")
        print(f"Successfully resized: {success_count}/{len(image_files)} PNG images")
        print(f"Results saved in: {self.output_dir}")
        print(f"Directory structure preserved")

def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))

    resizer = ImageResizer(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        target_width=args.width,
        target_height=args.height,
        method=args.method
    )

    resizer.process_all_images()

if __name__ == "__main__":
    main()
