from huggingface_hub import snapshot_download

REPO_ID = "huanngzh/Ouroboros3D-SVD-LGM"
snapshot_download(repo_id=REPO_ID, local_dir="checkpoint")
