import os
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

N_RANDOM_SAMPLES = 120_000
BUFFER_SIZE = 1_000_000

dataset = load_dataset(
    "allenai/c4",
    "en",
    cache_dir=os.getenv("HF_DATASETS_CACHE"),
    streaming=True
)

dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, seed=42)

samples = []
for i, example in enumerate(dataset["train"]):
    samples.append(example)

    if i + 1 >= N_RANDOM_SAMPLES:
        break

dataset_small = DatasetDict({"train": Dataset.from_list(samples)})

dataset_small.save_to_disk("./data/allenai___c4_120k")