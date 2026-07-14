# TinyStories Dataset

## Source

The dataset is sourced from Karpathy's cleaned GPT-4 version of TinyStories:

```
https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
```

## Downloading the Data (Colab)

The first 5,000 stories are downloaded from the raw parquet file and saved locally as `tinystories_5k.jsonl`:

```python
import pandas as pd, json

url = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet"
df = pd.read_parquet(url)

with open("tinystories_5k.jsonl", "w") as f:
    for story in df['text'].iloc[:5000]:
        f.write(json.dumps({"text": story}) + "\n")

print(f"Saved {len(df['text'].iloc[:5000])} stories")
```

This produces a **JSONL file** (one JSON object per line), which allows efficient loading of any number of stories without reading the entire file.

## File Format: `tinystories_5k.jsonl`

Each line is a JSON object with a single `text` key:

```json
{"text": "Once upon a time, there was a little girl named Lily..."}
{"text": "Tom had a big red ball. He liked to play with it..."}
```

- 5,000 lines total
- ~4MB on disk
- Supports partial loading — read only the first `n` lines without loading the rest

## Usage in `tinystories_dataset.py`

The `load_tinystories(num_stories, context_size)` function reads exactly `num_stories` lines from the JSONL file and prepares the data for a **character-level language model**:

```python
from tinystories_dataset import load_tinystories

inputs, targets, idx_to_char, encoded, vocab_size = load_tinystories(
    num_stories=500,
    context_size=4
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_stories` | `500` | Number of stories to load (max 5000) |
| `context_size` | `4` | Input sequence length for the character model |

### Returns

| Value | Type | Description |
|---|---|---|
| `inputs` | `torch.Tensor` | Shape `(N, context_size)` — input character sequences |
| `targets` | `torch.Tensor` | Shape `(N,)` — next character for each input |
| `idx_to_char` | `dict` | Maps integer id → character |
| `encoded` | `list` | Full text as integer sequence |
| `vocab_size` | `int` | Number of unique characters |

> **Note:** When `context_size=1`, `inputs` and `targets` are returned as empty lists. Use `encoded` directly in this case.
