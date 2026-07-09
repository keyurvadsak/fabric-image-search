from huggingface_hub import HfApi
import os

def upload():
    # Read the token you saved in token.txt
    with open("token.txt", "r") as f:
        token = f.read().strip()

    repo_id = "keyurvadsak6/fabric-image-search"
    api = HfApi()

    print("Uploading files to Hugging Face Space using the official API (this bypasses Git timeouts!)...")
    
    # Upload everything in the current directory except git and venv
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=[".git/*", "venv/*", "__pycache__/*", "token.txt"]
    )
    
    print("Upload 100% complete!")

if __name__ == "__main__":
    upload()
