"""
WikiText-2 dataset (lightweight local pretraining data)
https://huggingface.co/datasets/Salesforce/wikitext
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:

python3 -u "fineweb copy.py"

Will clean the caching, then save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
dataset_repo = "Salesforce/wikitext"
dataset_config = "wikitext-2-raw-v1"
local_dir = "edu_fineweb10B"  # kept for train_gpt2.py compatibility
shard_size = int(1e6)  # 1M tokens per shard, suitable for local machines

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    text = doc["text"].strip()
    if not text:
        return np.array([], dtype=np.uint16)
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 **
                                       16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def process_split(dataset_split, out_split, nprocs):
    hf_endpoint = os.environ.get(
        "HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    data_url = (
        f"{hf_endpoint}/datasets/{dataset_repo}/resolve/main/"
        f"{dataset_config}/{dataset_split}-00000-of-00001.parquet"
    )
    print(f"Loading {data_url}")
    dataset = load_dataset(
        "parquet",
        data_files={dataset_split: data_url},
        split=dataset_split,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Loaded {len(dataset):,} rows for split={dataset_split}.")
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if len(tokens) == 0:
                continue
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"{out_split} shard {shard_index}")

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                filename = os.path.join(
                    DATA_CACHE_DIR, f"wikitext2_{out_split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count +
                              remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                progress_bar.close()
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        filename = os.path.join(
            DATA_CACHE_DIR, f"wikitext2_{out_split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        progress_bar.close()


def main():
    # tokenize all documents and write output shards
    nprocs = max(1, os.cpu_count()//2)
    print(f"Saving token shards to {DATA_CACHE_DIR}")
    print(f"Using {nprocs} tokenizer worker processes")
    process_split("validation", "val", nprocs)
    process_split("train", "train", nprocs)
    print("Done.")


if __name__ == "__main__":
    main()
