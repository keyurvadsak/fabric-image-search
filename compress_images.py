import os
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compress_image(img_path):
    try:
        with Image.open(img_path) as img:
            # Resize image to max 512x512 while keeping aspect ratio
            img.thumbnail((512, 512))
            # Overwrite the original image with high compression
            img.save(img_path, format="JPEG", quality=75, optimize=True)
    except Exception as e:
        pass

def main():
    img_dir = Path("Images")
    images = list(img_dir.glob("*.jpg"))
    print(f"Compressing {len(images)} images to fix the 1GB limit...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(compress_image, images), total=len(images)))
        
    print("Compression complete!")

if __name__ == "__main__":
    main()
