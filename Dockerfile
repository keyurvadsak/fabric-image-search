FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV/FAISS if needed
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU version explicitly (saves ~2GB of space)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Start the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
