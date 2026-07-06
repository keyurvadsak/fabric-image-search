import open_clip
from torchvision import models

print("Downloading ResNet50 weights...")
models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

print("Downloading OpenCLIP weights...")
open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device="cpu"
)
print("All weights downloaded and cached successfully!")
