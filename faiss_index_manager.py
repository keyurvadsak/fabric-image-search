"""
FAISS index manager for textile image similarity search.

Manages TWO separate indices:
  1. ResNet index (2048-d) — for image-to-image similarity search
  2. CLIP index (512-d) — for text-to-image search

Handles building, saving, loading, and querying both indices.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from embeddings import (
    get_image_embedding,
    get_clip_image_embedding,
    get_resnet_embedding_dim,
    get_clip_embedding_dim,
)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Default storage paths
DEFAULT_STORE_DIR = Path("faiss_store")
RESNET_INDEX_FILENAME = "resnet.index"
CLIP_INDEX_FILENAME = "clip.index"
METADATA_FILENAME = "metadata.json"


def build_index(image_dir: str | Path) -> tuple:
    """
    Scan a directory of images and build BOTH FAISS indices:
      - ResNet index for image-to-image similarity
      - CLIP index for text-to-image search

    Args:
        image_dir: Path to directory containing textile images.

    Returns:
        (resnet_index, clip_index, image_paths)
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect all image file paths
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[faiss_index] Found {len(image_paths)} images in '{image_dir}'")

    resnet_dim = get_resnet_embedding_dim()  # 2048
    clip_dim = get_clip_embedding_dim()      # 512

    resnet_index = faiss.IndexFlatIP(resnet_dim)
    clip_index = faiss.IndexFlatIP(clip_dim)

    resnet_embeddings = []
    clip_embeddings = []
    valid_paths = []
    failed = []

    for i, img_path in enumerate(tqdm(image_paths, desc="Generating embeddings")):
        try:
            resnet_emb = get_image_embedding(img_path)
            clip_emb = get_clip_image_embedding(img_path)
            resnet_embeddings.append(resnet_emb)
            clip_embeddings.append(clip_emb)
            valid_paths.append(str(img_path))
        except Exception as e:
            print(f"  [!] Failed to process {img_path.name}: {e}")
            failed.append(str(img_path))

    if not valid_paths:
        raise RuntimeError("All images failed to process!")

    # Build both FAISS indices
    resnet_matrix = np.array(resnet_embeddings, dtype=np.float32)
    clip_matrix = np.array(clip_embeddings, dtype=np.float32)

    resnet_index.add(resnet_matrix)
    clip_index.add(clip_matrix)

    print(f"[faiss_index] Indices built: {resnet_index.ntotal} vectors, {len(failed)} failures")
    print(f"  ResNet index: {resnet_dim}-d, {resnet_index.ntotal} vectors")
    print(f"  CLIP index: {clip_dim}-d, {clip_index.ntotal} vectors")

    return resnet_index, clip_index, valid_paths


def save_index(
    resnet_index,
    clip_index,
    image_paths: list[str],
    store_dir: str | Path = DEFAULT_STORE_DIR,
):
    """Save both FAISS indices and image path metadata to disk."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    resnet_path = store_dir / RESNET_INDEX_FILENAME
    clip_path = store_dir / CLIP_INDEX_FILENAME
    meta_path = store_dir / METADATA_FILENAME

    faiss.write_index(resnet_index, str(resnet_path))
    faiss.write_index(clip_index, str(clip_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_paths": image_paths,
            "total": len(image_paths),
            "resnet_dim": resnet_index.d,
            "clip_dim": clip_index.d,
        }, f, indent=2)

    print(f"[faiss_index] Saved ResNet index -> {resnet_path}")
    print(f"[faiss_index] Saved CLIP index -> {clip_path}")
    print(f"[faiss_index] Saved metadata -> {meta_path}")


def load_index(store_dir: str | Path = DEFAULT_STORE_DIR) -> tuple:
    """
    Load both FAISS indices and metadata from disk.

    Returns:
        (resnet_index, clip_index, image_paths)
    """
    store_dir = Path(store_dir)
    resnet_path = store_dir / RESNET_INDEX_FILENAME
    clip_path = store_dir / CLIP_INDEX_FILENAME
    meta_path = store_dir / METADATA_FILENAME

    if not resnet_path.exists():
        raise FileNotFoundError(f"ResNet index not found at {resnet_path}")
    if not clip_path.exists():
        raise FileNotFoundError(f"CLIP index not found at {clip_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    resnet_index = faiss.read_index(str(resnet_path))
    clip_index = faiss.read_index(str(clip_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_paths = metadata["image_paths"]
    print(f"[faiss_index] Loaded dual index with {resnet_index.ntotal} vectors from {store_dir}")
    print(f"  ResNet: {resnet_index.d}-d | CLIP: {clip_index.d}-d")

    return resnet_index, clip_index, image_paths


def search(
    index,
    image_paths: list[str],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Search a FAISS index for the top-K most similar images.

    Args:
        index: A FAISS index (either ResNet or CLIP).
        image_paths: Ordered list of image paths matching the index.
        query_embedding: A 1-D numpy array — the query vector.
        top_k: Number of results to return.

    Returns:
        List of dicts: [{"path": str, "filename": str, "score": float, "rank": int}, ...]
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)

    scores, indices = index.search(query, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        results.append({
            "rank": rank,
            "path": image_paths[idx],
            "filename": str(image_paths[idx]).replace("\\", "/").split("/")[-1],
            "score": round(float(score), 4),
        })

    return results
