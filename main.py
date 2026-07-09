"""
FastAPI backend for textile image similarity search.

Dual-index architecture:
  - Image search -> ResNet50 index (texture/pattern/color focused)
  - Text search  -> CLIP index (semantic text-image matching)

Endpoints:
    GET  /              -- Serve the frontend UI
    POST /search        -- Upload an image, get top-5 similar textiles (ResNet)
    POST /search-text   -- Text query, get top-5 matching textiles (CLIP)
    GET  /images/{path} -- Serve original textile images for display
"""

import io
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image

from embeddings import get_image_embedding, get_text_embedding
from faiss_index_manager import load_index, search

# ── App setup ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Textile Similarity Search",
    description="Upload a fabric image or enter a text query to find the most similar textiles.",
    version="3.0.0",
)

# ── Load FAISS indices on startup ───────────────────────────────────────────────

STORE_DIR = Path("faiss_store")
IMAGE_BASE_DIR = Path("Images")

resnet_index = None
clip_index = None
image_paths = None


@app.on_event("startup")
async def startup_load_index():
    """Load both FAISS indices and metadata when the server starts."""
    global resnet_index, clip_index, image_paths
    try:
        resnet_index, clip_index, image_paths = load_index(STORE_DIR)
        print(f"[search_app] Dual index loaded: {resnet_index.ntotal} vectors")
    except FileNotFoundError as e:
        print(f"[search_app] [!] Index not found: {e}")
        print("[search_app] Run `python index_builder.py` first to build the indices.")


# ── Routes ──────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = Path("static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
):
    """
    Upload a fabric image and return the top-K most similar textiles.
    Uses ResNet50 index for visual similarity (texture, pattern, color).
    """
    if resnet_index is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded. Run `python index_builder.py` first.",
        )

    # Validate the uploaded file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Generate ResNet embedding for the query image
    query_embedding = get_image_embedding(image)

    # Search the ResNet FAISS index
    results = search(resnet_index, image_paths, query_embedding, top_k=top_k)

    # Add image URLs for the frontend to display
    for r in results:
        r["image_url"] = f"/images/{r['filename']}"

    return JSONResponse(content={
        "query_filename": file.filename,
        "search_type": "image",
        "top_k": top_k,
        "total_indexed": resnet_index.ntotal,
        "results": results,
    })


@app.post("/search-text")
async def search_by_text(
    query: str = Form(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
):
    """
    Search for textiles using a natural language text query.
    Uses CLIP index for semantic text-image matching.
    """
    if clip_index is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded. Run `python index_builder.py` first.",
        )

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Please provide a text query.")

    query_text = query.strip()

    # Generate CLIP text embedding
    query_embedding = get_text_embedding(query_text)

    # Search the CLIP FAISS index
    results = search(clip_index, image_paths, query_embedding, top_k=top_k)

    # Add image URLs for the frontend to display
    for r in results:
        r["image_url"] = f"/images/{r['filename']}"

    return JSONResponse(content={
        "query_text": query_text,
        "search_type": "text",
        "top_k": top_k,
        "total_indexed": clip_index.ntotal,
        "results": results,
    })


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a textile image from the images directory."""
    safe_name = Path(filename).name
    image_path = IMAGE_BASE_DIR / safe_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {safe_name}")

    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/stats")
async def get_stats():
    """Return index statistics."""
    if resnet_index is None:
        return {"status": "not_loaded", "total_indexed": 0}
    return {
        "status": "loaded",
        "total_indexed": resnet_index.ntotal,
        "resnet_dim": resnet_index.d,
        "clip_dim": clip_index.d if clip_index else 0,
    }


# ── Run ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)





    
