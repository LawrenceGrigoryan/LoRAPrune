from huggingface_hub import snapshot_download

data = snapshot_download(repo_id="MBZUAI/LaMini-instruction", repo_type="dataset", local_dir="./data/MBZUAI_LaMini-instruction", local_dir_use_symlinks=False)
