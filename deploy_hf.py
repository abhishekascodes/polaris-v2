#!/usr/bin/env python3
"""
Deploy OpenENV to HuggingFace Spaces
=====================================

Steps to deploy:

1. Install HF CLI:
   pip install huggingface_hub

2. Login to HuggingFace:
   huggingface-cli login

3. Create a new Space on huggingface.co:
   - Go to https://huggingface.co/new-space
   - Name: openenv-policy-engine
   - SDK: Docker
   - Visibility: Public

4. Clone and push:
   git clone https://huggingface.co/spaces/YOUR_USERNAME/openenv-policy-engine
   cd openenv-policy-engine

   # Copy all project files
   cp -r /path/to/openenv/* .

   # Push to HF Spaces
   git add .
   git commit -m "Deploy OpenENV Policy Engine"
   git push

The Dockerfile is already configured for port 7860 (HF Spaces default).
The server/app.py FastAPI app will be served automatically.

--- OR use this script to automate the push ---
"""

import subprocess
import sys
import os

def deploy(username: str, space_name: str = "openenv-policy-engine"):
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  OpenENV -> HuggingFace Spaces Deployment")
    print("=" * 60)

    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        print("  Install huggingface_hub first: pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{username}/{space_name}"
    print(f"\n  Target: https://huggingface.co/spaces/{repo_id}")

    # Create space if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"  Space created/verified: {repo_id}")
    except Exception as e:
        print(f"  Error creating space: {e}")
        sys.exit(1)

    # Upload all files
    print(f"\n  Uploading files from {project_dir}...")

    # Files to exclude from upload
    exclude = {".git", "__pycache__", "outputs", "uv.lock", ".dockerignore",
               "node_modules", ".env", "deploy_hf.py"}

    api.upload_folder(
        folder_path=project_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["*.pyc", "__pycache__/*", ".git/*", "outputs/*",
                         "uv.lock", "*.egg-info/*"],
    )

    print(f"\n  Deployed successfully!")
    print(f"  URL: https://huggingface.co/spaces/{repo_id}")
    print(f"\n  The Space will build automatically using your Dockerfile.")
    print(f"  Health check: https://{username}-{space_name}.hf.space/health")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy_hf.py YOUR_HF_USERNAME [space-name]")
        print("Example: python deploy_hf.py abhishekas openenv-policy-engine")
        sys.exit(1)

    username = sys.argv[1]
    space_name = sys.argv[2] if len(sys.argv) > 2 else "openenv-policy-engine"
    deploy(username, space_name)
