from huggingface_hub import snapshot_download

data = snapshot_download(repo_id="meta-llama/Llama-3.2-1B", repo_type="model", local_dir="./models/meta-llama_Llama-3.2-1B", local_dir_use_symlinks=False)
