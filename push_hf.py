#!/usr/bin/env python3
"""Push POLARIS v2 to HuggingFace Spaces."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import HfApi, create_repo

USERNAME = "asabhishek"
SPACE_NAME = "polaris-v2"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to upload (no model weights, no outputs, no test scripts)
INCLUDE = [
    "server/", "static/",
    "__init__.py", "models.py", "openenv.yaml", "pyproject.toml",
    "requirements.txt", "Dockerfile", ".dockerignore",
    "inference.py", "main.py", "client.py",
    "dashboard_server.py", "dashboard.html",
    "train_trl.py", "llm_benchmark.py",
    "rl_agent.py", "episode_logger.py",
    "README.md", "BLOG.md", "LICENSE",
]

EXCLUDE_PATTERNS = [
    "__pycache__", ".git", "outputs/", "*.pyc",
    "uv.lock", ".env", "check_collapse.py",
    "mega_test_*", "nuclear_test*", "smoke_test*",
    "validation_suite*", "evaluate_full*", "completion_check*",
    "ultimate_validation*", "generate_dashboard*", "deploy_hf*",
]

def main():
    api = HfApi()
    
    # Create Space
    print(f"Creating Space: {REPO_ID}...")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print(f"  ✓ Space created/exists: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"  Note: {e}")

    # Upload
    print(f"\nUploading project files...")
    api.upload_folder(
        folder_path=PROJECT_DIR,
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=EXCLUDE_PATTERNS,
        commit_message="POLARIS v2: Multi-Agent AI Governance Engine — Grand Finale",
    )
    
    print(f"\n  ✅ DEPLOYED: https://huggingface.co/spaces/{REPO_ID}")
    print(f"  Space will build automatically (2-3 minutes)")

if __name__ == "__main__":
    main()
