"""
CLI script to build BOTH FAISS indices from a directory of textile images.

Builds:
  1. ResNet index (2048-d) — for image-to-image similarity
  2. CLIP index (512-d) — for text-to-image search

Usage:
    python index_builder.py --image_dir ./images
    python index_builder.py --image_dir ./images --store_dir ./faiss_store
"""

import argparse
import time
from pathlib import Path

from faiss_index_manager import build_index, save_index


def main():
    parser = argparse.ArgumentParser(
        description="Build dual FAISS indices (ResNet + CLIP) from textile images."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./images",
        help="Path to the directory containing textile images (default: ./images)",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default="./faiss_store",
        help="Path to save the FAISS indices and metadata (default: ./faiss_store)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    store_dir = Path(args.store_dir)

    print("=" * 60)
    print("  Textile Image Indexer - Dual Index (ResNet + CLIP)")
    print("=" * 60)
    print(f"  Image directory : {image_dir.resolve()}")
    print(f"  Store directory  : {store_dir.resolve()}")
    print("  ResNet (2048-d)  : image-to-image similarity")
    print("  CLIP (512-d)     : text-to-image search")
    print("=" * 60)

    start = time.time()

    # Step 1: Build both indices
    resnet_index, clip_index, image_paths = build_index(image_dir)

    # Step 2: Save to disk
    save_index(resnet_index, clip_index, image_paths, store_dir)

    elapsed = time.time() - start
    print(f"\n[OK] Done! Indexed {len(image_paths)} images in {elapsed:.1f}s")
    print(f"   Index saved to: {store_dir.resolve()}")


if __name__ == "__main__":
    main()
