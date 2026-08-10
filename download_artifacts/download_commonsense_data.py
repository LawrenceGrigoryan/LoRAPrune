import os

from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CACHE_DIR = os.getenv("HF_BENCHMARKS_CACHE", "./data/benchmarks")

BENCHMARKS = [
    # (repo, config or None -> download all configs, trust_remote_code)
    ("cais/mmlu", None, False),
    ("Rowan/hellaswag", None, True),
    ("allenai/winogrande", "winogrande_xl", True),
]


def download(repo: str, config: str | None, trust_remote_code: bool) -> None:
    configs = [config] if config is not None else get_dataset_config_names(repo)
    logger.info(f"{repo}: {len(configs)} config(s) to download")
    for i, cfg in enumerate(configs, start=1):
        logger.info(f"[{i}/{len(configs)}] {repo}:{cfg}")
        ds = load_dataset(
            repo,
            cfg,
            cache_dir=CACHE_DIR,
            trust_remote_code=trust_remote_code,
        )
        for split, split_ds in ds.items():
            logger.info(f"  {split}: {len(split_ds)} examples")


for repo, config, trust in BENCHMARKS:
    download(repo, config, trust)

logger.info(f"Done. Cached under {CACHE_DIR}")
