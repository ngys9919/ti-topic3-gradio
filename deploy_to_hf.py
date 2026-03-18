from huggingface_hub import HfApi
import os

from dotenv import load_dotenv

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    print("Error: HF_TOKEN not found in .env file.")
    exit(1)

api = HfApi(token=token)

# Get the username to construct the repo_id
username = api.whoami()["name"]
repo_id = f"{username}/residual-network-trainer"

print(f"Deploying to Space: {repo_id}...")

# Create the Space
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)

# Upload README.md (Space configuration)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
)

# Upload the trainer file as app.py
api.upload_file(
    path_or_fileobj="resnet_trainer.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space",
)

# Upload the requirements.txt
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="space",
)

print(f"Successfully deployed! Visit: https://huggingface.co/spaces/{repo_id}")
