"""
Dual embedding extractor for textile similarity search.

- ResNet50 (ImageNet): Used for image-to-image search.
  Extracts visual features (texture, pattern, color, shape) from the avgpool layer.
  Produces 2048-d embeddings that excel at visual similarity matching.

- OpenCLIP ViT-B-32: Used for text-to-image search only.
  Encodes text queries into the same vector space as CLIP image embeddings.

Singleton pattern ensures models are loaded only once.
"""

import torch
import torch.nn as nn
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from torchvision import models, transforms

# ── Singleton model holders ─────────────────────────────────────────────────────

# ResNet50 for image-to-image similarity
_resnet_model = None
_resnet_preprocess = None

# OpenCLIP for text-to-image search
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None

_device = None


def _get_device():
    """Get and cache the compute device."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_resnet():
    """Load ResNet50 as a feature extractor (once)."""
    global _resnet_model, _resnet_preprocess

    if _resnet_model is not None:
        return

    device = _get_device()
    print(f"[embeddings] Loading ResNet50 feature extractor on {device}...")

    # Load pretrained ResNet50 and remove the classification head
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Use everything up to and including avgpool (output: 2048-d)
    _resnet_model = nn.Sequential(*list(base_model.children())[:-1])  # remove fc layer
    _resnet_model = _resnet_model.to(device)
    _resnet_model.eval()

    # Standard ImageNet preprocessing
    _resnet_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("[embeddings] ResNet50 loaded successfully.")


def _load_clip():
    """Load OpenCLIP model and tokenizer (once)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer

    if _clip_model is not None:
        return

    device = _get_device()
    print(f"[embeddings] Loading OpenCLIP ViT-B-32 on {device}...")

    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip_model.eval()
    print("[embeddings] OpenCLIP loaded successfully.")


# ── Public API ──────────────────────────────────────────────────────────────────


def get_image_embedding(image_input) -> np.ndarray:
    """
    Convert a single image to a normalized 2048-d embedding vector using ResNet50.
    Focuses on visual features: texture, pattern, color, and shape.

    Args:
        image_input: Either a file path (str / Path) or a PIL.Image.Image object.

    Returns:
        np.ndarray of shape (2048,), L2-normalized.
    """
    _load_resnet()

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected str, Path, or PIL.Image, got {type(image_input)}")

    device = _get_device()
    image_tensor = _resnet_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = _resnet_model(image_tensor)

    # Flatten from (1, 2048, 1, 1) to (2048,)
    embedding = embedding.flatten()

    # L2 normalize so we can use inner product as cosine similarity
    embedding = embedding / embedding.norm()

    return embedding.cpu().to(torch.float32).numpy()


def get_clip_image_embedding(image_input) -> np.ndarray:
    """
    Convert a single image to a normalized 512-d CLIP embedding.
    Used internally for building the CLIP index (for text search).

    Args:
        image_input: Either a file path (str / Path) or a PIL.Image.Image object.

    Returns:
        np.ndarray of shape (512,), L2-normalized.
    """
    _load_clip()

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected str, Path, or PIL.Image, got {type(image_input)}")

    device = _get_device()
    image_tensor = _clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = _clip_model.encode_image(image_tensor)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().to(torch.float32).numpy().flatten()


def get_text_embedding(text: str) -> np.ndarray:
    """
    Convert a text query to a normalized 512-d CLIP embedding vector.
    Used for text-to-image search against the CLIP FAISS index.

    Args:
        text: A natural language description, e.g. "red floral silk fabric"

    Returns:
        np.ndarray of shape (512,), L2-normalized.
    """
    _load_clip()

    device = _get_device()
    tokens = _clip_tokenizer([text]).to(device)

    with torch.no_grad():
        embedding = _clip_model.encode_text(tokens)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().to(torch.float32).numpy().flatten()


def get_resnet_embedding_dim() -> int:
    """Return dimensionality of ResNet50 embeddings (2048)."""
    return 2048


def get_clip_embedding_dim() -> int:
    """Return dimensionality of CLIP embeddings (512)."""
    return 512
