import os

from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CACHE_DIR = os.getenv("HF_BENCHMARKS_CACHE", "./data/benchmarks")

BENCHMARKS = [
    # (repo, config or None, trust_remote_code)
    ("google/IFEval", None, False),
]


def download(repo: str, config: str | None, trust_remote_code: bool) -> None:
    logger.info(f"{repo}: config={config}")
    ds = load_dataset(
        repo,
        config,
        cache_dir=CACHE_DIR,
        trust_remote_code=trust_remote_code,
    )
    for split, split_ds in ds.items():
        logger.info(f"  {split}: {len(split_ds)} examples")


for repo, config, trust in BENCHMARKS:
    download(repo, config, trust)

logger.info(f"Done. Cached under {CACHE_DIR}")
