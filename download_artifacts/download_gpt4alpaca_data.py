import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("vicgalle/alpaca-gpt4", "default", cache_dir=os.getenv("HF_DATASETS_CACHE"))