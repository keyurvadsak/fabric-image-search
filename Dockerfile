FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV/FAISS if needed
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU version explicitly (saves ~2GB of space)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements, ignoring the massive CUDA versions of torch
RUN grep -ivE "^torch|^torchvision" requirements.txt > req-slim.txt && pip install -r req-slim.txt

# Copy the rest of the application
COPY . .

# Start the application using python so it automatically reads the PORT environment variable
CMD ["python", "main.py"]
