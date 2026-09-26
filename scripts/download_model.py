"""Download a GGUF model from Hugging Face into models/ with resume support.

Usage: .venv/Scripts/python.exe scripts/download_model.py <repo_id> <filename>
Requires HF_HOME to point at the shared cache (config/auto_config.py pins it).
"""
import sys

from huggingface_hub import hf_hub_download


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo_id, filename = sys.argv[1], sys.argv[2]
    print(f"downloading {repo_id}/{filename} ...", flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models")
    print(f"DONE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
