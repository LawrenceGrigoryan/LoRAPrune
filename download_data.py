import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("allenai/c4", "en", cache_dir=os.getenv("HF_DATASETS_CACHE"))